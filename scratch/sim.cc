#include "sim.h"

NS_LOG_COMPONENT_DEFINE("PPOTrafficSim");
NS_OBJECT_ENSURE_REGISTERED(TrafficGymEnv);

TypeId TrafficGymEnv::GetTypeId() {
    static TypeId tid = TypeId("ns3::TrafficGymEnv")
        .SetParent<OpenGymEnv>()
        .SetGroupName("OpenGym")
        .AddConstructor<TrafficGymEnv>();
    return tid;
}

TrafficGymEnv::TrafficGymEnv() { NS_LOG_FUNCTION(this); }

TrafficGymEnv::TrafficGymEnv(SimConfig cfg) : m_cfg(cfg) {
    NS_LOG_FUNCTION(this);
    m_stepCount  = 0;
    m_avgDelay   = 0.0;
    m_throughput = 0.0;
    m_packetLoss = 0.0;
    m_queueLen.resize(cfg.numNodes, 0.0);
    m_linkUtil.resize(cfg.numNodes, 0.0);
    m_delay.resize(cfg.numNodes, 0.0);

    if (cfg.topoType == "linear")      BuildLinearTopology();
    else if (cfg.topoType == "grid")   BuildGridTopology();
    else                               BuildRandomTopology();

    m_flowMonitor = m_flowHelper.InstallAll();
}

TrafficGymEnv::~TrafficGymEnv() { NS_LOG_FUNCTION(this); }

void TrafficGymEnv::ScheduleNextStep() {
    Simulator::Schedule(Seconds(m_cfg.stepInterval),
                        &TrafficGymEnv::Step, this);
}

void TrafficGymEnv::Step() {
    Notify();
    if (!GetGameOver()) ScheduleNextStep();
}

Ptr<OpenGymSpace> TrafficGymEnv::GetObservationSpace() {
    uint32_t obsSize = m_cfg.numNodes * OBS_PER_NODE;
    std::vector<uint32_t> shape = {obsSize};
    return CreateObject<OpenGymBoxSpace>(0.0, 1.0, shape,
                                        TypeNameGet<float>());
}

Ptr<OpenGymSpace> TrafficGymEnv::GetActionSpace() {
    uint32_t actionSize = m_cfg.numNodes * 2;
    std::vector<uint32_t> shape = {actionSize};
    float maxVal = (float)(std::max(m_cfg.numNodes,
                   (uint32_t)MAX_BW_LEVELS) - 1);
    return CreateObject<OpenGymBoxSpace>(0.0, maxVal, shape,
                                        TypeNameGet<uint32_t>());
}

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

float TrafficGymEnv::GetReward() {
    double normDelay = std::min(m_avgDelay / 1.0, 1.0);
    double normTput  = std::min(m_throughput / 1e6, 1.0);
    double normLoss  = std::min(m_packetLoss, 1.0);
    return (float)(
        - m_cfg.delayWeight * normDelay
        + m_cfg.tputWeight  * normTput
        - m_cfg.lossWeight  * normLoss
    );
}

bool TrafficGymEnv::GetGameOver() {
    return (Simulator::Now().GetSeconds() >= m_cfg.simTime);
}

std::string TrafficGymEnv::GetExtraInfo() {
    std::ostringstream oss;
    oss << "step="  << m_stepCount
        << ",delay="<< m_avgDelay
        << ",tput=" << m_throughput
        << ",loss=" << m_packetLoss;
    return oss.str();
}

bool TrafficGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action) {
    Ptr<OpenGymBoxContainer<uint32_t>> act =
        DynamicCast<OpenGymBoxContainer<uint32_t>>(action);
    if (!act) return false;

    for (uint32_t i = 0; i < m_cfg.numNodes; i++) {
        uint32_t nextHop = act->GetValue(i * 2)     % m_cfg.numNodes;
        uint32_t bwLevel = act->GetValue(i * 2 + 1) % MAX_BW_LEVELS;

        if (i < m_devices.GetN()) {
            Ptr<PointToPointNetDevice> dev =
                DynamicCast<PointToPointNetDevice>(m_devices.Get(i));
            if (dev) dev->SetDataRate(DataRate(m_bwLevels[bwLevel]));
        }

        Ptr<Ipv4> ipv4 = m_nodes.Get(i)->GetObject<Ipv4>();
        if (!ipv4) continue;
        Ptr<Ipv4StaticRouting> route =
            Ipv4RoutingHelper::GetRouting<Ipv4StaticRouting>(
                ipv4->GetRoutingProtocol());
        if (route && nextHop != i && nextHop < m_cfg.numNodes) {
            Ipv4Address nhAddr = m_interfaces.GetAddress(nextHop);
            if (route->GetNRoutes() > 0) route->RemoveRoute(0);
            route->SetDefaultRoute(nhAddr, 1);
        }
    }
    m_stepCount++;
    return true;
}

void TrafficGymEnv::CollectStats() {
    m_flowMonitor->CheckForLostPackets();
    FlowMonitor::FlowStatsContainer stats = m_flowMonitor->GetFlowStats();

    double totalDelay = 0.0, totalTput = 0.0, totalLoss = 0.0;
    uint32_t flowCount = 0;
    double now = Simulator::Now().GetSeconds();

    for (auto& kv : stats) {
        auto& s = kv.second;
        if (s.rxPackets > 0) {
            totalDelay += s.delaySum.GetSeconds() / s.rxPackets;
            if (now > 0) totalTput += s.rxBytes * 8.0 / now;
        }
        if (s.txPackets > 0)
            totalLoss += (double)(s.txPackets - s.rxPackets) / s.txPackets;
        flowCount++;
    }

    if (flowCount > 0) {
        m_avgDelay   = totalDelay / flowCount;
        m_throughput = totalTput  / flowCount;
        m_packetLoss = totalLoss  / flowCount;
    }

    for (uint32_t i = 0; i < m_cfg.numNodes; i++) {
        m_queueLen[i] = std::min(m_avgDelay * 10.0, 1.0);
        m_linkUtil[i] = std::min(m_throughput / 1e6, 1.0);
        m_delay[i]    = std::min(m_avgDelay, 1.0);
    }
}

// ── Linear topology ───────────────────────────────────────────
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
        NetDeviceContainer lnk =
            p2p.Install(m_nodes.Get(i), m_nodes.Get(i + 1));
        m_devices.Add(lnk);
        m_interfaces.Add(addr.Assign(lnk));
        addr.NewNetwork();
    }

    uint16_t port = 9;
    ApplicationContainer srvApp =
        UdpServerHelper(port).Install(m_nodes.Get(m_cfg.numNodes - 1));
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
    NS_LOG_UNCOND("Linear topology built: " << m_cfg.numNodes << " nodes");
}

// ── Grid topology ─────────────────────────────────────────────
void TrafficGymEnv::BuildGridTopology() {
    uint32_t side = (uint32_t)std::sqrt((double)m_cfg.numNodes);
    if (side * side != m_cfg.numNodes) {
        NS_LOG_WARN("numNodes not perfect square, using linear");
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

    for (uint32_t r = 0; r < side; r++) {
        for (uint32_t c = 0; c < side; c++) {
            uint32_t cur = r * side + c;
            if (c + 1 < side) {
                NetDeviceContainer lnk = p2p.Install(
                    m_nodes.Get(cur), m_nodes.Get(r * side + c + 1));
                m_devices.Add(lnk);
                m_interfaces.Add(addr.Assign(lnk));
                addr.NewNetwork();
            }
            if (r + 1 < side) {
                NetDeviceContainer lnk = p2p.Install(
                    m_nodes.Get(cur), m_nodes.Get((r+1)*side + c));
                m_devices.Add(lnk);
                m_interfaces.Add(addr.Assign(lnk));
                addr.NewNetwork();
            }
        }
    }

    uint16_t port = 9;
    ApplicationContainer srvApp =
        UdpServerHelper(port).Install(m_nodes.Get(m_cfg.numNodes - 1));
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
    NS_LOG_UNCOND("Grid topology built: " << side << "x" << side);
}

// ── Random topology ───────────────────────────────────────────
void TrafficGymEnv::BuildRandomTopology() {
    m_nodes.Create(m_cfg.numNodes);
    InternetStackHelper internet;
    internet.Install(m_nodes);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue(m_cfg.dataRate));
    p2p.SetChannelAttribute("Delay",   StringValue(m_cfg.delay));

    Ipv4AddressHelper addr;
    addr.SetBase("10.1.1.0", "255.255.255.0");

    // Backbone chain for guaranteed connectivity
    for (uint32_t i = 0; i < m_cfg.numNodes - 1; i++) {
        NetDeviceContainer lnk =
            p2p.Install(m_nodes.Get(i), m_nodes.Get(i + 1));
        m_devices.Add(lnk);
        m_interfaces.Add(addr.Assign(lnk));
        addr.NewNetwork();
    }

    // Extra random links
    Ptr<UniformRandomVariable> rng = CreateObject<UniformRandomVariable>();
    for (uint32_t k = 0; k < m_cfg.numNodes / 2; k++) {
        uint32_t a = rng->GetInteger(0, m_cfg.numNodes - 1);
        uint32_t b = rng->GetInteger(0, m_cfg.numNodes - 1);
        if (a != b && std::abs((int)a - (int)b) > 1) {
            NetDeviceContainer lnk =
                p2p.Install(m_nodes.Get(a), m_nodes.Get(b));
            m_devices.Add(lnk);
            m_interfaces.Add(addr.Assign(lnk));
            addr.NewNetwork();
        }
    }

    uint16_t port = 9;
    ApplicationContainer srvApp =
        UdpServerHelper(port).Install(m_nodes.Get(m_cfg.numNodes - 1));
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
    NS_LOG_UNCOND("Random topology built: " << m_cfg.numNodes << " nodes");
}

// ── Main ──────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    SimConfig cfg;

    CommandLine cmd;
    cmd.AddValue("numNodes",     "Number of nodes",        cfg.numNodes);
    cmd.AddValue("topoType",     "Topology type",          cfg.topoType);
    cmd.AddValue("simTime",      "Simulation time (s)",    cfg.simTime);
    cmd.AddValue("port",         "OpenGym ZMQ port",       cfg.openGymPort);
    cmd.AddValue("dataRate",     "Link data rate",         cfg.dataRate);
    cmd.AddValue("delay",        "Link delay",             cfg.delay);
    cmd.AddValue("stepInterval", "Step interval (s)",      cfg.stepInterval);
    cmd.Parse(argc, argv);

    NS_LOG_UNCOND("Starting PPO Traffic Sim"
        << " | nodes="    << cfg.numNodes
        << " | topo="     << cfg.topoType
        << " | time="     << cfg.simTime  << "s"
        << " | port="     << cfg.openGymPort
        << " | step="     << cfg.stepInterval << "s");

    Ptr<TrafficGymEnv> env = CreateObject<TrafficGymEnv>(cfg);
    Ptr<OpenGymInterface> openGym =
        CreateObject<OpenGymInterface>(cfg.openGymPort);
    env->SetOpenGymInterface(openGym);
    env->ScheduleNextStep();

    Simulator::Stop(Seconds(cfg.simTime));
    Simulator::Run();
    openGym->NotifySimulationEnd();
    Simulator::Destroy();

    NS_LOG_UNCOND("Simulation complete.");
    return 0;
}
