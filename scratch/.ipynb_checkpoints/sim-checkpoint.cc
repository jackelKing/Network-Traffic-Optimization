// =============================================================
// PPO Network Traffic Optimization — NS3 Simulation
// Controls: next-hop routing + bandwidth allocation
// Reward:   delay + throughput + packet loss (combined)
// =============================================================

#include "sim.h"

NS_LOG_COMPONENT_DEFINE("PPOTrafficSim");

// ─── TypeId ──────────────────────────────────────────────────
TypeId TrafficGymEnv::GetTypeId() {
    static TypeId tid = TypeId("TrafficGymEnv")
        .SetParent<OpenGymEnv>()
        .SetGroupName("OpenGym");
    return tid;
}

// ─── Constructor ─────────────────────────────────────────────
TrafficGymEnv::TrafficGymEnv(SimConfig cfg) : m_cfg(cfg) {
    m_stepCount = 0;
    m_done      = false;
    m_avgDelay  = 0.0;
    m_throughput= 0.0;
    m_packetLoss= 0.0;

    m_queueLen.resize(cfg.numNodes, 0.0);
    m_linkUtil.resize(cfg.numNodes, 0.0);
    m_delay.resize(cfg.numNodes, 0.0);

    // Build topology based on config
    if (cfg.topoType == "linear")      BuildLinearTopology();
    else if (cfg.topoType == "grid")   BuildGridTopology();
    else                               BuildRandomTopology();

    // Start flow monitor
    m_flowMonitor = m_flowHelper.InstallAll();
}

TrafficGymEnv::~TrafficGymEnv() {}

// ─── Observation space ────────────────────────────────────────
// Flat vector: [queue_len, link_util, delay] × numNodes
// All values normalised to [0, 1]
Ptr<OpenGymSpace> TrafficGymEnv::GetObservationSpace() {
    uint32_t obsSize = m_cfg.numNodes * OBS_PER_NODE;
    std::vector<uint32_t> shape = {obsSize};
    std::string dtype = TypeNameGet<float>();
    Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(
        0.0, 1.0, shape, dtype);
    return space;
}

// ─── Action space ─────────────────────────────────────────────
// Discrete tuple: [next_hop_0..N, bw_level_0..N]
// Each node picks: next hop (0..numNodes-1) + bw level (0..4)
Ptr<OpenGymSpace> TrafficGymEnv::GetActionSpace() {
    uint32_t actionSize = m_cfg.numNodes * 2; // next_hop + bw per node
    std::vector<uint32_t> shape = {actionSize};
    std::string dtype = TypeNameGet<uint32_t>();
    // next_hop: 0..numNodes-1, bw_level: 0..MAX_BW_LEVELS-1
    // We flatten both into one discrete space of size numNodes*2
    Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(
        0.0, (float)(std::max(m_cfg.numNodes, (uint32_t)MAX_BW_LEVELS) - 1),
        shape, dtype);
    return space;
}

// ─── Observation ──────────────────────────────────────────────
Ptr<OpenGymDataContainer> TrafficGymEnv::GetObservation() {
    CollectStats();

    uint32_t obsSize = m_cfg.numNodes * OBS_PER_NODE;
    std::vector<uint32_t> shape = {obsSize};
    Ptr<OpenGymBoxContainer<float>> obs =
        CreateObject<OpenGymBoxContainer<float>>(shape);

    for (uint32_t i = 0; i < m_cfg.numNodes; i++) {
        obs->AddValue((float)m_queueLen[i]);
        obs->AddValue((float)m_linkUtil[i]);
        obs->AddValue((float)m_delay[i]);
    }
    return obs;
}

// ─── Reward ───────────────────────────────────────────────────
float TrafficGymEnv::GetReward() {
    // Normalise components to [0,1] then combine with weights
    // Higher throughput = good, higher delay/loss = bad
    double normDelay  = std::min(m_avgDelay / 1.0, 1.0);   // cap at 1s
    double normTput   = std::min(m_throughput / 1e6, 1.0);  // cap at 1Mbps
    double normLoss   = std::min(m_packetLoss, 1.0);

    float reward = (float)(
        - m_cfg.delayWeight * normDelay
        + m_cfg.tputWeight  * normTput
        - m_cfg.lossWeight  * normLoss
    );
    return reward;
}

// ─── Game over ────────────────────────────────────────────────
bool TrafficGymEnv::GetGameOver() {
    return (Simulator::Now().GetSeconds() >= m_cfg.simTime);
}

// ─── Extra info ───────────────────────────────────────────────
std::string TrafficGymEnv::GetExtraInfo() {
    std::ostringstream oss;
    oss << "step=" << m_stepCount
        << ",delay=" << m_avgDelay
        << ",tput="  << m_throughput
        << ",loss="  << m_packetLoss;
    return oss.str();
}

// ─── Execute actions ──────────────────────────────────────────
bool TrafficGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action) {
    Ptr<OpenGymBoxContainer<uint32_t>> act =
        DynamicCast<OpenGymBoxContainer<uint32_t>>(action);

    for (uint32_t i = 0; i < m_cfg.numNodes; i++) {
        uint32_t nextHop  = act->GetValue(i * 2)     % m_cfg.numNodes;
        uint32_t bwLevel  = act->GetValue(i * 2 + 1) % MAX_BW_LEVELS;

        // Apply next-hop: update static routing table
        Ptr<Ipv4>            ipv4   = m_nodes.Get(i)->GetObject<Ipv4>();
        Ptr<Ipv4StaticRouting> route =
            Ipv4RoutingHelper::GetRouting<Ipv4StaticRouting>(
                ipv4->GetRoutingProtocol());

        if (route && nextHop != i) {
            // Remove old default route, add new one via chosen next hop
            route->RemoveRoute(0);
            Ipv4Address nhAddr =
                m_interfaces.GetAddress(nextHop);
            route->SetDefaultRoute(nhAddr, 1);
        }

        // Apply bandwidth: update point-to-point link rate
        if (i < m_devices.GetN()) {
            Ptr<PointToPointNetDevice> dev =
                DynamicCast<PointToPointNetDevice>(m_devices.Get(i));
            if (dev) {
                dev->SetDataRate(DataRate(m_bwLevels[bwLevel]));
            }
        }
    }

    m_stepCount++;
    return true;
}

// ─── Collect stats from FlowMonitor ───────────────────────────
void TrafficGymEnv::CollectStats() {
    m_flowMonitor->CheckForLostPackets();
    FlowMonitor::FlowStatsContainer stats =
        m_flowMonitor->GetFlowStats();

    double totalDelay = 0.0, totalTput = 0.0, totalLoss = 0.0;
    uint32_t flowCount = 0;

    for (auto& kv : stats) {
        auto& s = kv.second;
        if (s.rxPackets > 0) {
            totalDelay += s.delaySum.GetSeconds() / s.rxPackets;
            totalTput  += s.rxBytes * 8.0 /
                          Simulator::Now().GetSeconds();
        }
        uint32_t sent = s.txPackets;
        uint32_t lost = sent - s.rxPackets;
        if (sent > 0) totalLoss += (double)lost / sent;
        flowCount++;
    }

    if (flowCount > 0) {
        m_avgDelay   = totalDelay / flowCount;
        m_throughput = totalTput  / flowCount;
        m_packetLoss = totalLoss  / flowCount;
    }

    // Simple queue/util estimates per node (normalised)
    for (uint32_t i = 0; i < m_cfg.numNodes; i++) {
        m_queueLen[i] = std::min(m_avgDelay * 10.0, 1.0);
        m_linkUtil[i] = std::min(m_throughput / 1e6, 1.0);
        m_delay[i]    = std::min(m_avgDelay, 1.0);
    }
}

// ─── Topology: Linear ─────────────────────────────────────────
void TrafficGymEnv::BuildLinearTopology() {
    m_nodes.Create(m_cfg.numNodes);

    InternetStackHelper internet;
    internet.Install(m_nodes);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue(m_cfg.dataRate));
    p2p.SetChannelAttribute("Delay",   StringValue(m_cfg.delay));

    Ipv4AddressHelper addr;
    addr.SetBase("10.1.1.0", "255.255.255.0");

    for (uint32_t i = 0; i < m_cfg.numNodes - 1; i++) {
        NetDeviceContainer link =
            p2p.Install(m_nodes.Get(i), m_nodes.Get(i + 1));
        m_devices.Add(link);
        Ipv4InterfaceContainer ifc = addr.Assign(link);
        m_interfaces.Add(ifc);
        addr.NewNetwork();
    }

    // UDP traffic: node 0 → last node
    uint16_t port = 9;
    UdpServerHelper server(port);
    ApplicationContainer srvApp =
        server.Install(m_nodes.Get(m_cfg.numNodes - 1));
    srvApp.Start(Seconds(0.0));
    srvApp.Stop(Seconds(m_cfg.simTime));

    UdpClientHelper client(
        m_interfaces.GetAddress(m_cfg.numNodes * 2 - 1), port);
    client.SetAttribute("MaxPackets", UintegerValue(100000));
    client.SetAttribute("Interval",   TimeValue(MilliSeconds(10)));
    client.SetAttribute("PacketSize", UintegerValue(1024));
    ApplicationContainer cliApp = client.Install(m_nodes.Get(0));
    cliApp.Start(Seconds(0.5));
    cliApp.Stop(Seconds(m_cfg.simTime));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
}

// ─── Topology: Grid ───────────────────────────────────────────
void TrafficGymEnv::BuildGridTopology() {
    // Grid: sqrt(numNodes) × sqrt(numNodes)
    // Falls back to linear if numNodes is not a perfect square
    uint32_t side = (uint32_t)std::sqrt((double)m_cfg.numNodes);
    if (side * side != m_cfg.numNodes) {
        NS_LOG_WARN("numNodes not a perfect square, falling back to linear");
        BuildLinearTopology();
        return;
    }

    m_nodes.Create(m_cfg.numNodes);
    InternetStackHelper internet;
    internet.Install(m_nodes);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue(m_cfg.dataRate));
    p2p.SetChannelAttribute("Delay",   StringValue(m_cfg.delay));

    Ipv4AddressHelper addr;
    addr.SetBase("10.1.1.0", "255.255.255.0");

    // Connect horizontal and vertical neighbours
    for (uint32_t r = 0; r < side; r++) {
        for (uint32_t c = 0; c < side; c++) {
            uint32_t cur = r * side + c;
            // right neighbour
            if (c + 1 < side) {
                uint32_t right = r * side + (c + 1);
                NetDeviceContainer link =
                    p2p.Install(m_nodes.Get(cur), m_nodes.Get(right));
                m_devices.Add(link);
                m_interfaces.Add(addr.Assign(link));
                addr.NewNetwork();
            }
            // bottom neighbour
            if (r + 1 < side) {
                uint32_t below = (r + 1) * side + c;
                NetDeviceContainer link =
                    p2p.Install(m_nodes.Get(cur), m_nodes.Get(below));
                m_devices.Add(link);
                m_interfaces.Add(addr.Assign(link));
                addr.NewNetwork();
            }
        }
    }

    // Traffic: top-left → bottom-right
    uint16_t port = 9;
    UdpServerHelper server(port);
    ApplicationContainer srvApp =
        server.Install(m_nodes.Get(m_cfg.numNodes - 1));
    srvApp.Start(Seconds(0.0));
    srvApp.Stop(Seconds(m_cfg.simTime));

    UdpClientHelper client(
        m_interfaces.GetAddress(m_interfaces.GetN() - 1), port);
    client.SetAttribute("MaxPackets", UintegerValue(100000));
    client.SetAttribute("Interval",   TimeValue(MilliSeconds(10)));
    client.SetAttribute("PacketSize", UintegerValue(1024));
    ApplicationContainer cliApp = client.Install(m_nodes.Get(0));
    cliApp.Start(Seconds(0.5));
    cliApp.Stop(Seconds(m_cfg.simTime));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
}

// ─── Topology: Random ─────────────────────────────────────────
void TrafficGymEnv::BuildRandomTopology() {
    // Random graph: each node connects to 2 random others
    // Ensures full connectivity via a base linear backbone
    m_nodes.Create(m_cfg.numNodes);
    InternetStackHelper internet;
    internet.Install(m_nodes);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue(m_cfg.dataRate));
    p2p.SetChannelAttribute("Delay",   StringValue(m_cfg.delay));

    Ipv4AddressHelper addr;
    addr.SetBase("10.1.1.0", "255.255.255.0");

    // Backbone: linear chain for guaranteed connectivity
    for (uint32_t i = 0; i < m_cfg.numNodes - 1; i++) {
        NetDeviceContainer link =
            p2p.Install(m_nodes.Get(i), m_nodes.Get(i + 1));
        m_devices.Add(link);
        m_interfaces.Add(addr.Assign(link));
        addr.NewNetwork();
    }

    // Extra random links (skip if already linked)
    Ptr<UniformRandomVariable> rng = CreateObject<UniformRandomVariable>();
    uint32_t extraLinks = m_cfg.numNodes / 2;
    for (uint32_t k = 0; k < extraLinks; k++) {
        uint32_t a = rng->GetInteger(0, m_cfg.numNodes - 1);
        uint32_t b = rng->GetInteger(0, m_cfg.numNodes - 1);
        if (a != b && std::abs((int)a - (int)b) > 1) {
            NetDeviceContainer link =
                p2p.Install(m_nodes.Get(a), m_nodes.Get(b));
            m_devices.Add(link);
            m_interfaces.Add(addr.Assign(link));
            addr.NewNetwork();
        }
    }

    // Traffic: node 0 → last node
    uint16_t port = 9;
    UdpServerHelper server(port);
    ApplicationContainer srvApp =
        server.Install(m_nodes.Get(m_cfg.numNodes - 1));
    srvApp.Start(Seconds(0.0));
    srvApp.Stop(Seconds(m_cfg.simTime));

    UdpClientHelper client(
        m_interfaces.GetAddress(m_interfaces.GetN() - 1), port);
    client.SetAttribute("MaxPackets", UintegerValue(100000));
    client.SetAttribute("Interval",   TimeValue(MilliSeconds(10)));
    client.SetAttribute("PacketSize", UintegerValue(1024));
    ApplicationContainer cliApp = client.Install(m_nodes.Get(0));
    cliApp.Start(Seconds(0.5));
    cliApp.Stop(Seconds(m_cfg.simTime));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
}

// ─── Main ─────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    SimConfig cfg;

    CommandLine cmd;
    cmd.AddValue("numNodes",  "Number of nodes",         cfg.numNodes);
    cmd.AddValue("topoType",  "Topology type",           cfg.topoType);
    cmd.AddValue("simTime",   "Simulation time (s)",     cfg.simTime);
    cmd.AddValue("port",      "OpenGym ZMQ port",        cfg.openGymPort);
    cmd.AddValue("dataRate",  "Link data rate",          cfg.dataRate);
    cmd.AddValue("delay",     "Link delay",              cfg.delay);
    cmd.Parse(argc, argv);

    NS_LOG_UNCOND("Starting PPO Traffic Sim | nodes=" << cfg.numNodes
        << " topo=" << cfg.topoType
        << " time=" << cfg.simTime << "s");

    Ptr<OpenGymInterface> openGym =
        CreateObject<OpenGymInterface>(cfg.openGymPort);

    Ptr<TrafficGymEnv> env = CreateObject<TrafficGymEnv>(cfg);
    env->SetOpenGymInterface(openGym);

    Simulator::Stop(Seconds(cfg.simTime));
    Simulator::Run();
    Simulator::Destroy();

    NS_LOG_UNCOND("Simulation complete.");
    return 0;
}
