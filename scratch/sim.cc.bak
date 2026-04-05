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
    m_stepCount    = 0;
    m_avgDelay     = 0.0;
    m_throughput   = 0.0;
    m_packetLoss   = 0.0;
    m_avgCongestion= 0.0;
    m_prevDelay    = 0.0;
    m_prevTput     = 0.0;
    m_prevLoss     = 0.0;

    m_queueLen.resize(cfg.numNodes, 0.0);
    m_linkUtil.resize(cfg.numNodes, 0.0);
    m_delay.resize(cfg.numNodes, 0.0);
    m_lossRate.resize(cfg.numNodes, 0.0);
    m_congestion.resize(cfg.numNodes, 0.0);
    m_staticRouting.resize(cfg.numNodes);

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

// ── Static routing ────────────────────────────────────────────
void TrafficGymEnv::InstallStaticRoutes() {
    Ipv4StaticRoutingHelper staticHelper;
    for (uint32_t i = 0; i < m_cfg.numNodes; i++) {
        Ptr<Ipv4> ipv4 = m_nodes.Get(i)->GetObject<Ipv4>();
        m_staticRouting[i] = staticHelper.GetStaticRouting(ipv4);
        uint32_t nextHop = std::min(i + 1, m_cfg.numNodes - 1);
        if (nextHop == i) continue;
        for (auto& lnk : m_links) {
            uint32_t other = 999;
            if (lnk.first  == i) other = lnk.second;
            if (lnk.second == i) other = lnk.first;
            if (other == nextHop) {
                Ptr<Ipv4> nhIpv4 = m_nodes.Get(nextHop)->GetObject<Ipv4>();
                Ipv4Address nhAddr = nhIpv4->GetAddress(1,0).GetLocal();
                m_staticRouting[i]->SetDefaultRoute(nhAddr, 1);
                break;
            }
        }
    }
    NS_LOG_UNCOND("Static routes installed: " << m_cfg.numNodes << " nodes");
}

void TrafficGymEnv::UpdateRoute(uint32_t nodeId, uint32_t nextHop) {
    if (nodeId >= m_cfg.numNodes || nextHop >= m_cfg.numNodes) return;
    if (nodeId == nextHop) return;
    if (!m_staticRouting[nodeId]) return;

    Ptr<Ipv4> ipv4    = m_nodes.Get(nodeId)->GetObject<Ipv4>();
    Ptr<Ipv4> nhIpv4  = m_nodes.Get(nextHop)->GetObject<Ipv4>();
    uint32_t  nIfaces = ipv4->GetNInterfaces();
    uint32_t  nhFaces = nhIpv4->GetNInterfaces();

    for (uint32_t iface = 1; iface < nIfaces; iface++) {
        Ipv4Address myNet = ipv4->GetAddress(iface,0).GetLocal();
        Ipv4Mask    mask  = ipv4->GetAddress(iface,0).GetMask();
        for (uint32_t ni = 1; ni < nhFaces; ni++) {
            Ipv4Address nhAddr = nhIpv4->GetAddress(ni,0).GetLocal();
            if (myNet.CombineMask(mask) == nhAddr.CombineMask(mask)) {
                if (m_staticRouting[nodeId]->GetNRoutes() > 0)
                    m_staticRouting[nodeId]->RemoveRoute(0);
                m_staticRouting[nodeId]->SetDefaultRoute(nhAddr, iface);
                return;
            }
        }
    }
}

// ── Observation space ─────────────────────────────────────────
// 5 metrics per node: queue, util, delay, loss, congestion
Ptr<OpenGymSpace> TrafficGymEnv::GetObservationSpace() {
    uint32_t obsSize = m_cfg.numNodes * OBS_PER_NODE;
    std::vector<uint32_t> shape = {obsSize};
    return CreateObject<OpenGymBoxSpace>(0.0, 1.0, shape,
                                        TypeNameGet<float>());
}

// ── Action space ──────────────────────────────────────────────
Ptr<OpenGymSpace> TrafficGymEnv::GetActionSpace() {
    uint32_t actionSize = m_cfg.numNodes * 2;
    std::vector<uint32_t> shape = {actionSize};
    float maxVal = (float)(std::max(m_cfg.numNodes,
                   (uint32_t)MAX_BW_LEVELS) - 1);
    return CreateObject<OpenGymBoxSpace>(0.0, maxVal, shape,
                                        TypeNameGet<uint32_t>());
}

// ── Observation ───────────────────────────────────────────────
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
        obs->AddValue((float)m_lossRate[i]);
        obs->AddValue((float)m_congestion[i]);
    }
    return obs;
}

// ── Reward ────────────────────────────────────────────────────
// Key: reward congestion avoidance — this is where PPO beats OSPF
float TrafficGymEnv::GetReward() {
    double normDelay = std::min(m_avgDelay / 1.0,    1.0);
    double normTput  = std::min(m_throughput / 1e6,  1.0);
    double normLoss  = std::min(m_packetLoss,        1.0);
    double normCong  = std::min(m_avgCongestion,     1.0);

    // Delta rewards — reward improvement over last step
    double delayDelta = m_prevDelay  - normDelay;  // positive = improved
    double tputDelta  = normTput  - m_prevTput;    // positive = improved
    double lossDelta  = m_prevLoss   - normLoss;   // positive = improved

    double baseReward =
        - m_cfg.delayWeight       * normDelay
        + m_cfg.tputWeight        * normTput
        - m_cfg.lossWeight        * normLoss
        - m_cfg.congestionWeight  * normCong;

    // CRITICAL: Heavy penalty for zero throughput
    // Prevents PPO from learning to avoid routing entirely
    double zeroTputPenalty = 0.0;
    if (normTput < 0.01)
        zeroTputPenalty = -0.3;
    else if (normTput < 0.05)
        zeroTputPenalty = -0.1;

    // Improvement bonus
    double improvementBonus = 0.1 * (delayDelta + tputDelta + lossDelta);

    // Congestion avoidance bonus ONLY when throughput is meaningful
    double congAvoidBonus = 0.0;
    if (normTput > 0.1 && normCong < 0.1)
        congAvoidBonus = 0.3;
    else if (normTput > 0.05 && normCong < 0.3)
        congAvoidBonus = 0.1;

    // Load balance bonus
    double maxUtil = *std::max_element(m_linkUtil.begin(), m_linkUtil.end());
    double minUtil = *std::min_element(m_linkUtil.begin(), m_linkUtil.end());
    double balanceBonus = 0.0;
    if (normTput > 0.05)  // only reward balance when traffic is flowing
        balanceBonus = 0.1 * (1.0 - (maxUtil - minUtil));

    m_prevDelay = normDelay;
    m_prevTput  = normTput;
    m_prevLoss  = normLoss;

    return (float)(baseReward + zeroTputPenalty +
                   improvementBonus + congAvoidBonus + balanceBonus);
}

bool TrafficGymEnv::GetGameOver() {
    return (Simulator::Now().GetSeconds() >= m_cfg.simTime);
}

std::string TrafficGymEnv::GetExtraInfo() {
    std::ostringstream oss;
    oss << "step="  << m_stepCount
        << ",delay="<< m_avgDelay
        << ",tput=" << m_throughput
        << ",loss=" << m_packetLoss
        << ",cong=" << m_avgCongestion;
    return oss.str();
}

// ── Execute actions ───────────────────────────────────────────
bool TrafficGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action) {
    Ptr<OpenGymBoxContainer<uint32_t>> act =
        DynamicCast<OpenGymBoxContainer<uint32_t>>(action);
    if (!act) return false;

    for (uint32_t i = 0; i < m_cfg.numNodes; i++) {
        uint32_t nextHop = act->GetValue(i * 2)     % m_cfg.numNodes;
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

// ── Collect stats ─────────────────────────────────────────────
void TrafficGymEnv::CollectStats() {
    m_flowMonitor->CheckForLostPackets();
    FlowMonitor::FlowStatsContainer stats = m_flowMonitor->GetFlowStats();

    double totalDelay = 0.0, totalTput  = 0.0;
    double totalLoss  = 0.0, totalCong  = 0.0;
    uint32_t flowCount = 0;
    double now = Simulator::Now().GetSeconds();

    // Reset per-node stats
    std::fill(m_queueLen.begin(),  m_queueLen.end(),  0.0);
    std::fill(m_linkUtil.begin(),  m_linkUtil.end(),  0.0);
    std::fill(m_delay.begin(),     m_delay.end(),     0.0);
    std::fill(m_lossRate.begin(),  m_lossRate.end(),  0.0);
    std::fill(m_congestion.begin(),m_congestion.end(),0.0);

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
            // Congestion = loss × delay (both high = congested)
            fCong = fLoss * std::min(fDelay * 10.0, 1.0);
        }

        totalDelay += fDelay;
        totalTput  += fTput;
        totalLoss  += fLoss;
        totalCong  += fCong;
        flowCount++;

        // Assign to node round-robin
        uint32_t nodeIdx = fi % m_cfg.numNodes;
        m_delay[nodeIdx]     = std::max(m_delay[nodeIdx],
                               std::min(fDelay, 1.0));
        m_linkUtil[nodeIdx]  = std::max(m_linkUtil[nodeIdx],
                               std::min(fTput / 1e6, 1.0));
        m_lossRate[nodeIdx]  = std::max(m_lossRate[nodeIdx],
                               std::min(fLoss, 1.0));
        m_congestion[nodeIdx]= std::max(m_congestion[nodeIdx],
                               std::min(fCong, 1.0));
        m_queueLen[nodeIdx]  = m_congestion[nodeIdx];
        fi++;
    }

    if (flowCount > 0) {
        m_avgDelay    = std::min(totalDelay / flowCount, 1.0);
        m_throughput  = totalTput  / flowCount;
        m_packetLoss  = std::min(totalLoss  / flowCount, 1.0);
        m_avgCongestion = std::min(totalCong / flowCount, 1.0);
    }
}

// ── Multiple flows — creates congestion OSPF cannot handle ────
void TrafficGymEnv::InstallMultipleFlows(uint32_t numFlows) {
    uint32_t n = m_nodes.GetN();

    for (uint32_t f = 0; f < numFlows; f++) {
        // Create competing flows across the network
        // Different src/dst pairs force traffic to share links
        uint32_t src, dst;
        if (f == 0) {
            src = 0; dst = n - 1;           // main flow: 0 → last
        } else if (f == 1) {
            src = 0; dst = n / 2;           // flow 2: 0 → middle
        } else if (f == 2) {
            src = 1; dst = n - 1;           // flow 3: 1 → last
        } else {
            src = f % (n / 2);
            dst = n - 1 - (f % (n / 2));
            if (src == dst) dst = (src + n/2) % n;
        }
        if (src >= n) src = 0;
        if (dst >= n) dst = n - 1;
        if (src == dst) continue;

        uint16_t port = 9 + f;

        // Server
        UdpServerHelper server(port);
        ApplicationContainer srvApp = server.Install(m_nodes.Get(dst));
        srvApp.Start(Seconds(0.0));
        srvApp.Stop(Seconds(m_cfg.simTime));

        // Get destination address
        // Find interface on dst node
        Ptr<Ipv4> dstIpv4 = m_nodes.Get(dst)->GetObject<Ipv4>();
        Ipv4Address dstAddr = dstIpv4->GetAddress(1, 0).GetLocal();

        // Client — vary intervals to create different congestion levels
        // f=0: 5ms (high load), f=1: 8ms, f=2: 12ms (lower load)
        // Aggressive traffic to create heavy congestion
        // OSPF cannot adapt — PPO learns to reroute around bottlenecks
        uint32_t intervalMs = 2 + (f * 2);  // 2ms, 4ms, 6ms — heavy load
        UdpClientHelper client(dstAddr, port);
        client.SetAttribute("MaxPackets", UintegerValue(1000000));
        client.SetAttribute("Interval",
                            TimeValue(MilliSeconds(intervalMs)));
        client.SetAttribute("PacketSize", UintegerValue(1024));  // larger packets
        ApplicationContainer cliApp = client.Install(m_nodes.Get(src));
        cliApp.Start(Seconds(0.1 + f * 0.05));
        cliApp.Stop(Seconds(m_cfg.simTime));

        NS_LOG_UNCOND("Flow " << f << ": node " << src
                      << " -> node " << dst
                      << " interval=" << intervalMs << "ms");
    }
}

// ── Topologies ────────────────────────────────────────────────
void TrafficGymEnv::BuildLinearTopology() {
    m_nodes.Create(m_cfg.numNodes);

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

    for (uint32_t i = 0; i < m_cfg.numNodes - 1; i++) {
        NetDeviceContainer lnk =
            p2p.Install(m_nodes.Get(i), m_nodes.Get(i + 1));
        m_devices.Add(lnk);
        m_interfaces.Add(addr.Assign(lnk));
        m_links.push_back({i, i + 1});
        addr.NewNetwork();
    }

    InstallStaticRoutes();

    // Multiple flows: numNodes/2 competing flows
    uint32_t numFlows = std::max(2u, m_cfg.numNodes / 2);
    InstallMultipleFlows(numFlows);

    NS_LOG_UNCOND("Linear topology: " << m_cfg.numNodes
                  << " nodes, " << numFlows << " flows");
}

void TrafficGymEnv::BuildGridTopology() {
    uint32_t side = (uint32_t)std::sqrt((double)m_cfg.numNodes);
    if (side * side != m_cfg.numNodes) {
        BuildLinearTopology();
        return;
    }

    m_nodes.Create(m_cfg.numNodes);

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

    for (uint32_t r = 0; r < side; r++) {
        for (uint32_t c = 0; c < side; c++) {
            uint32_t cur = r * side + c;
            if (c + 1 < side) {
                uint32_t right = r * side + c + 1;
                NetDeviceContainer lnk =
                    p2p.Install(m_nodes.Get(cur), m_nodes.Get(right));
                m_devices.Add(lnk);
                m_interfaces.Add(addr.Assign(lnk));
                m_links.push_back({cur, right});
                addr.NewNetwork();
            }
            if (r + 1 < side) {
                uint32_t below = (r + 1) * side + c;
                NetDeviceContainer lnk =
                    p2p.Install(m_nodes.Get(cur), m_nodes.Get(below));
                m_devices.Add(lnk);
                m_interfaces.Add(addr.Assign(lnk));
                m_links.push_back({cur, below});
                addr.NewNetwork();
            }
        }
    }

    InstallStaticRoutes();

    uint32_t numFlows = std::max(3u, m_cfg.numNodes / 3);
    InstallMultipleFlows(numFlows);

    NS_LOG_UNCOND("Grid topology: " << side << "x" << side
                  << ", " << numFlows << " flows");
}

void TrafficGymEnv::BuildRandomTopology() {
    m_nodes.Create(m_cfg.numNodes);

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

    for (uint32_t i = 0; i < m_cfg.numNodes - 1; i++) {
        NetDeviceContainer lnk =
            p2p.Install(m_nodes.Get(i), m_nodes.Get(i + 1));
        m_devices.Add(lnk);
        m_interfaces.Add(addr.Assign(lnk));
        m_links.push_back({i, i + 1});
        addr.NewNetwork();
    }

    Ptr<UniformRandomVariable> rng = CreateObject<UniformRandomVariable>();
    for (uint32_t k = 0; k < m_cfg.numNodes / 2; k++) {
        uint32_t a = rng->GetInteger(0, m_cfg.numNodes - 1);
        uint32_t b = rng->GetInteger(0, m_cfg.numNodes - 1);
        if (a != b && std::abs((int)a - (int)b) > 1) {
            NetDeviceContainer lnk =
                p2p.Install(m_nodes.Get(a), m_nodes.Get(b));
            m_devices.Add(lnk);
            m_interfaces.Add(addr.Assign(lnk));
            m_links.push_back({a, b});
            addr.NewNetwork();
        }
    }

    InstallStaticRoutes();

    uint32_t numFlows = std::max(4u, m_cfg.numNodes / 3);
    InstallMultipleFlows(numFlows);

    NS_LOG_UNCOND("Random topology: " << m_cfg.numNodes
                  << " nodes, " << numFlows << " flows");
}

int main(int argc, char* argv[]) {
    SimConfig cfg;
    CommandLine cmd;
    cmd.AddValue("numNodes",     "Number of nodes",     cfg.numNodes);
    cmd.AddValue("topoType",     "Topology type",       cfg.topoType);
    cmd.AddValue("simTime",      "Sim time (s)",        cfg.simTime);
    cmd.AddValue("port",         "OpenGym ZMQ port",    cfg.openGymPort);
    cmd.AddValue("dataRate",     "Link data rate",      cfg.dataRate);
    cmd.AddValue("delay",        "Link delay",          cfg.delay);
    cmd.AddValue("stepInterval", "Step interval (s)",   cfg.stepInterval);
    cmd.Parse(argc, argv);

    NS_LOG_UNCOND("Starting PPO Traffic Sim"
        << " | nodes="  << cfg.numNodes
        << " | topo="   << cfg.topoType
        << " | time="   << cfg.simTime << "s"
        << " | port="   << cfg.openGymPort
        << " | step="   << cfg.stepInterval << "s");

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
