#include "mesh_3x3_sim.h"

NS_LOG_COMPONENT_DEFINE("Mesh3x3Sim");
NS_OBJECT_ENSURE_REGISTERED(Mesh3x3Env);

TypeId Mesh3x3Env::GetTypeId() {
    static TypeId tid = TypeId("ns3::Mesh3x3Env")
        .SetParent<OpenGymEnv>()
        .SetGroupName("OpenGym")
        .AddConstructor<Mesh3x3Env>();
    return tid;
}

Mesh3x3Env::Mesh3x3Env() { NS_LOG_FUNCTION(this); }

Mesh3x3Env::Mesh3x3Env(MeshConfig cfg) : m_cfg(cfg) {
    NS_LOG_FUNCTION(this);
    m_stepCount     = 0;
    m_avgDelay      = 0.0;
    m_throughput    = 0.0;
    m_packetLoss    = 0.0;
    m_avgCongestion = 0.0;
    m_prevDelay     = 0.0;
    m_prevTput      = 0.0;
    m_prevLoss      = 0.0;

    m_queueLen.resize(NUM_NODES, 0.0);
    m_linkUtil.resize(NUM_NODES, 0.0);
    m_delay.resize(NUM_NODES, 0.0);
    m_lossRate.resize(NUM_NODES, 0.0);
    m_congestion.resize(NUM_NODES, 0.0);
    m_staticRouting.resize(NUM_NODES);

    BuildMesh3x3Topology();
    m_flowMonitor = m_flowHelper.InstallAll();
}

Mesh3x3Env::~Mesh3x3Env() { NS_LOG_FUNCTION(this); }

void Mesh3x3Env::ScheduleNextStep() {
    Simulator::Schedule(Seconds(m_cfg.stepInterval),
                        &Mesh3x3Env::Step, this);
}

void Mesh3x3Env::Step() {
    Notify();
    if (!GetGameOver()) ScheduleNextStep();
}

// ── Build 3x3 Mesh Topology ───────────────────────────────────
void Mesh3x3Env::BuildMesh3x3Topology() {
    m_nodes.Create(NUM_NODES);

    Ipv4StaticRoutingHelper staticHelper;
    Ipv4ListRoutingHelper   listHelper;
    listHelper.Add(staticHelper, 10);
    InternetStackHelper internet;
    internet.SetRoutingHelper(listHelper);
    internet.Install(m_nodes);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue(m_cfg.dataRate));
    p2p.SetChannelAttribute("Delay",   StringValue(m_cfg.delay));

    Ipv4AddressHelper addr;
    addr.SetBase("10.1.1.0", "255.255.255.0");

    // Create 3x3 mesh with horizontal and vertical links
    // Node numbering:
    // 0 - 1 - 2
    // |   |   |
    // 3 - 4 - 5
    // |   |   |
    // 6 - 7 - 8

    for (uint32_t row = 0; row < MESH_SIZE; row++) {
        for (uint32_t col = 0; col < MESH_SIZE; col++) {
            uint32_t cur = row * MESH_SIZE + col;

            // Horizontal link (to the right)
            if (col + 1 < MESH_SIZE) {
                uint32_t right = row * MESH_SIZE + col + 1;
                NetDeviceContainer link = p2p.Install(
                    m_nodes.Get(cur), m_nodes.Get(right));
                m_devices.Add(link);
                m_interfaces.Add(addr.Assign(link));
                m_links.push_back({cur, right});
                addr.NewNetwork();
            }

            // Vertical link (downward)
            if (row + 1 < MESH_SIZE) {
                uint32_t below = (row + 1) * MESH_SIZE + col;
                NetDeviceContainer link = p2p.Install(
                    m_nodes.Get(cur), m_nodes.Get(below));
                m_devices.Add(link);
                m_interfaces.Add(addr.Assign(link));
                m_links.push_back({cur, below});
                addr.NewNetwork();
            }
        }
    }

    InstallStaticRoutes();
    InstallMeshTraffic();

    NS_LOG_UNCOND("3x3 Mesh topology created: " 
                  << NUM_NODES << " nodes, "
                  << m_links.size() << " links");
}

// ── Install Traffic Flows ─────────────────────────────────────
void Mesh3x3Env::InstallMeshTraffic() {
    // Create multiple traffic flows across the mesh
    // This creates congestion that requires intelligent routing
    
    struct Flow {
        uint32_t src, dst;
        uint32_t intervalMs;
    };

    std::vector<Flow> flows = {
        {0, 8, 5},   // Corner to corner (main diagonal)
        {2, 6, 6},   // Other diagonal
        {0, 2, 7},   // Top row
        {0, 6, 8},   // Left column
        {2, 8, 9},   // Right column
        {4, 0, 10},  // Center to corner
        {4, 8, 11},  // Center to opposite corner
    };

    for (size_t f = 0; f < flows.size(); f++) {
        uint32_t src = flows[f].src;
        uint32_t dst = flows[f].dst;
        uint32_t interval = flows[f].intervalMs;
        uint16_t port = 9000 + f;

        // Server
        UdpServerHelper server(port);
        ApplicationContainer srvApp = server.Install(m_nodes.Get(dst));
        srvApp.Start(Seconds(0.0));
        srvApp.Stop(Seconds(m_cfg.simTime));

        // Get destination address
        Ptr<Ipv4> dstIpv4 = m_nodes.Get(dst)->GetObject<Ipv4>();
        Ipv4Address dstAddr = dstIpv4->GetAddress(1, 0).GetLocal();

        // Client
        UdpClientHelper client(dstAddr, port);
        client.SetAttribute("MaxPackets", UintegerValue(1000000));
        client.SetAttribute("Interval", 
                          TimeValue(MilliSeconds(interval)));
        client.SetAttribute("PacketSize", UintegerValue(512));
        
        ApplicationContainer cliApp = client.Install(m_nodes.Get(src));
        cliApp.Start(Seconds(0.1 + f * 0.05));
        cliApp.Stop(Seconds(m_cfg.simTime));

        NS_LOG_UNCOND("Flow " << f << ": node " << src 
                      << " -> node " << dst 
                      << " (" << interval << "ms)");
    }
}

// ── Static Routing ────────────────────────────────────────────
void Mesh3x3Env::InstallStaticRoutes() {
    Ipv4StaticRoutingHelper staticHelper;
    for (uint32_t i = 0; i < NUM_NODES; i++) {
        Ptr<Ipv4> ipv4 = m_nodes.Get(i)->GetObject<Ipv4>();
        m_staticRouting[i] = staticHelper.GetStaticRouting(ipv4);
        
        // Initial routing: prefer right/down movement toward destination
        uint32_t nextHop = std::min(i + 1, NUM_NODES - 1);
        if (nextHop == i) continue;
        
        for (auto& lnk : m_links) {
            uint32_t other = 999;
            if (lnk.first == i)  other = lnk.second;
            if (lnk.second == i) other = lnk.first;
            if (other == nextHop) {
                Ptr<Ipv4> nhIpv4 = m_nodes.Get(nextHop)->GetObject<Ipv4>();
                Ipv4Address nhAddr = nhIpv4->GetAddress(1, 0).GetLocal();
                m_staticRouting[i]->SetDefaultRoute(nhAddr, 1);
                break;
            }
        }
    }
    NS_LOG_UNCOND("Static routes installed for " << NUM_NODES << " nodes");
}

void Mesh3x3Env::UpdateRoute(uint32_t nodeId, uint32_t nextHop) {
    if (nodeId >= NUM_NODES || nextHop >= NUM_NODES) return;
    if (nodeId == nextHop) return;
    if (!m_staticRouting[nodeId]) return;

    Ptr<Ipv4> ipv4   = m_nodes.Get(nodeId)->GetObject<Ipv4>();
    Ptr<Ipv4> nhIpv4 = m_nodes.Get(nextHop)->GetObject<Ipv4>();
    uint32_t nIfaces = ipv4->GetNInterfaces();
    uint32_t nhFaces = nhIpv4->GetNInterfaces();

    for (uint32_t iface = 1; iface < nIfaces; iface++) {
        Ipv4Address myNet = ipv4->GetAddress(iface, 0).GetLocal();
        Ipv4Mask mask = ipv4->GetAddress(iface, 0).GetMask();
        for (uint32_t ni = 1; ni < nhFaces; ni++) {
            Ipv4Address nhAddr = nhIpv4->GetAddress(ni, 0).GetLocal();
            if (myNet.CombineMask(mask) == nhAddr.CombineMask(mask)) {
                if (m_staticRouting[nodeId]->GetNRoutes() > 0)
                    m_staticRouting[nodeId]->RemoveRoute(0);
                m_staticRouting[nodeId]->SetDefaultRoute(nhAddr, iface);
                return;
            }
        }
    }
}

// ── Observation Space ─────────────────────────────────────────
Ptr<OpenGymSpace> Mesh3x3Env::GetObservationSpace() {
    uint32_t obsSize = NUM_NODES * OBS_PER_NODE;
    std::vector<uint32_t> shape = {obsSize};
    return CreateObject<OpenGymBoxSpace>(0.0, 1.0, shape,
                                        TypeNameGet<float>());
}

// ── Action Space ──────────────────────────────────────────────
Ptr<OpenGymSpace> Mesh3x3Env::GetActionSpace() {
    uint32_t actionSize = NUM_NODES * 2;
    std::vector<uint32_t> shape = {actionSize};
    float maxVal = (float)(std::max(NUM_NODES, 
                   (uint32_t)MAX_BW_LEVELS) - 1);
    return CreateObject<OpenGymBoxSpace>(0.0, maxVal, shape,
                                        TypeNameGet<uint32_t>());
}

// ── Get Observation ───────────────────────────────────────────
Ptr<OpenGymDataContainer> Mesh3x3Env::GetObservation() {
    CollectStats();
    uint32_t obsSize = NUM_NODES * OBS_PER_NODE;
    std::vector<uint32_t> shape = {obsSize};
    Ptr<OpenGymBoxContainer<float>> obs = 
        CreateObject<OpenGymBoxContainer<float>>(shape);
    
    for (uint32_t i = 0; i < NUM_NODES; i++) {
        obs->AddValue((float)m_queueLen[i]);
        obs->AddValue((float)m_linkUtil[i]);
        obs->AddValue((float)m_delay[i]);
        obs->AddValue((float)m_lossRate[i]);
        obs->AddValue((float)m_congestion[i]);
    }
    return obs;
}

// ── Reward Function ───────────────────────────────────────────
float Mesh3x3Env::GetReward() {
    double normDelay = std::min(m_avgDelay / 1.0, 1.0);
    double normTput  = std::min(m_throughput / 1e6, 1.0);
    double normLoss  = std::min(m_packetLoss, 1.0);
    double normCong  = std::min(m_avgCongestion, 1.0);

    // Delta rewards
    double delayDelta = m_prevDelay - normDelay;
    double tputDelta  = normTput - m_prevTput;
    double lossDelta  = m_prevLoss - normLoss;

    double baseReward = 
        - m_cfg.delayWeight      * normDelay
        + m_cfg.tputWeight       * normTput
        - m_cfg.lossWeight       * normLoss
        - m_cfg.congestionWeight * normCong;

    double improvementBonus = 0.1 * (delayDelta + tputDelta + lossDelta);

    // Mesh-specific: reward using multiple paths
    double pathDiversityBonus = 0.0;
    if (m_linkUtil.size() > 0) {
        double maxUtil = *std::max_element(m_linkUtil.begin(), 
                                           m_linkUtil.end());
        double minUtil = *std::min_element(m_linkUtil.begin(), 
                                           m_linkUtil.end());
        // Reward balanced utilization across mesh
        pathDiversityBonus = 0.15 * (1.0 - (maxUtil - minUtil));
    }

    // Congestion avoidance bonus
    double congAvoidBonus = 0.0;
    if (normCong < 0.1 && normTput > 0.5)
        congAvoidBonus = 0.2;
    else if (normCong < 0.3 && normTput > 0.3)
        congAvoidBonus = 0.1;

    m_prevDelay = normDelay;
    m_prevTput  = normTput;
    m_prevLoss  = normLoss;

    return (float)(baseReward + improvementBonus + 
                   congAvoidBonus + pathDiversityBonus);
}

bool Mesh3x3Env::GetGameOver() {
    return (Simulator::Now().GetSeconds() >= m_cfg.simTime);
}

std::string Mesh3x3Env::GetExtraInfo() {
    std::ostringstream oss;
    oss << "step=" << m_stepCount
        << ",delay=" << m_avgDelay
        << ",tput=" << m_throughput
        << ",loss=" << m_packetLoss
        << ",cong=" << m_avgCongestion;
    return oss.str();
}

// ── Execute Actions ───────────────────────────────────────────
bool Mesh3x3Env::ExecuteActions(Ptr<OpenGymDataContainer> action) {
    Ptr<OpenGymBoxContainer<uint32_t>> act = 
        DynamicCast<OpenGymBoxContainer<uint32_t>>(action);
    if (!act) return false;

    for (uint32_t i = 0; i < NUM_NODES; i++) {
        uint32_t nextHop = act->GetValue(i * 2) % NUM_NODES;
        uint32_t bwLevel = act->GetValue(i * 2 + 1) % MAX_BW_LEVELS;

        // Apply bandwidth
        if (i * 2 < m_devices.GetN()) {
            Ptr<PointToPointNetDevice> dev = 
                DynamicCast<PointToPointNetDevice>(m_devices.Get(i * 2));
            if (dev) dev->SetDataRate(DataRate(m_bwLevels[bwLevel]));
        }

        // Apply routing
        UpdateRoute(i, nextHop);
    }
    m_stepCount++;
    return true;
}

// ── Collect Statistics ────────────────────────────────────────
void Mesh3x3Env::CollectStats() {
    m_flowMonitor->CheckForLostPackets();
    FlowMonitor::FlowStatsContainer stats = m_flowMonitor->GetFlowStats();

    double totalDelay = 0.0, totalTput = 0.0;
    double totalLoss = 0.0, totalCong = 0.0;
    uint32_t flowCount = 0;
    double now = Simulator::Now().GetSeconds();

    std::fill(m_queueLen.begin(), m_queueLen.end(), 0.0);
    std::fill(m_linkUtil.begin(), m_linkUtil.end(), 0.0);
    std::fill(m_delay.begin(), m_delay.end(), 0.0);
    std::fill(m_lossRate.begin(), m_lossRate.end(), 0.0);
    std::fill(m_congestion.begin(), m_congestion.end(), 0.0);

    uint32_t fi = 0;
    for (auto& kv : stats) {
        auto& s = kv.second;
        double fDelay = 0.0, fTput = 0.0, fLoss = 0.0, fCong = 0.0;

        if (s.rxPackets > 0) {
            fDelay = s.delaySum.GetSeconds() / s.rxPackets;
            if (now > 0) fTput = s.rxBytes * 8.0 / now;
        }
        if (s.txPackets > 0) {
            fLoss = (double)(s.txPackets - s.rxPackets) / s.txPackets;
            fCong = fLoss * std::min(fDelay * 10.0, 1.0);
        }

        totalDelay += fDelay;
        totalTput  += fTput;
        totalLoss  += fLoss;
        totalCong  += fCong;
        flowCount++;

        uint32_t nodeIdx = fi % NUM_NODES;
        m_delay[nodeIdx]      = std::max(m_delay[nodeIdx], 
                                        std::min(fDelay, 1.0));
        m_linkUtil[nodeIdx]   = std::max(m_linkUtil[nodeIdx], 
                                        std::min(fTput / 1e6, 1.0));
        m_lossRate[nodeIdx]   = std::max(m_lossRate[nodeIdx], 
                                        std::min(fLoss, 1.0));
        m_congestion[nodeIdx] = std::max(m_congestion[nodeIdx], 
                                        std::min(fCong, 1.0));
        m_queueLen[nodeIdx]   = m_congestion[nodeIdx];
        fi++;
    }

    if (flowCount > 0) {
        m_avgDelay      = std::min(totalDelay / flowCount, 1.0);
        m_throughput    = totalTput / flowCount;
        m_packetLoss    = std::min(totalLoss / flowCount, 1.0);
        m_avgCongestion = std::min(totalCong / flowCount, 1.0);
    }
}

// ── Main ──────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    MeshConfig cfg;
    CommandLine cmd;
    cmd.AddValue("simTime", "Simulation time (s)", cfg.simTime);
    cmd.AddValue("port", "OpenGym ZMQ port", cfg.openGymPort);
    cmd.AddValue("dataRate", "Link data rate", cfg.dataRate);
    cmd.AddValue("delay", "Link delay", cfg.delay);
    cmd.AddValue("stepInterval", "Step interval (s)", cfg.stepInterval);
    cmd.Parse(argc, argv);

    NS_LOG_UNCOND("=== 3x3 Mesh PPO Training ==="
        << "\n  Nodes:    " << NUM_NODES
        << "\n  Mesh:     " << MESH_SIZE << "x" << MESH_SIZE
        << "\n  SimTime:  " << cfg.simTime << "s"
        << "\n  Port:     " << cfg.openGymPort
        << "\n  DataRate: " << cfg.dataRate
        << "\n  Delay:    " << cfg.delay
        << "\n  Step:     " << cfg.stepInterval << "s");

    Ptr<Mesh3x3Env> env = CreateObject<Mesh3x3Env>(cfg);
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
