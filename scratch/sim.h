#pragma once

#include "ns3/opengym-module.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv4-list-routing-helper.h"
#include "ns3/queue-disc.h"
#include "ns3/traffic-control-helper.h"

#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <map>
#include <utility>

using namespace ns3;

// OBS per node: queue_len, link_util, delay, loss_rate, congestion
#define MAX_BW_LEVELS 5
#define OBS_PER_NODE  5

struct SimConfig {
    uint32_t    numNodes         = 4;
    std::string topoType         = "linear";
    double      simTime          = 10.0;
    uint16_t    openGymPort      = 5555;
    std::string dataRate         = "10Mbps";
    std::string delay            = "2ms";
    double      delayWeight      = 0.3;
    double      tputWeight       = 0.4;
    double      lossWeight       = 0.15;
    double      congestionWeight = 0.15;
    double      stepInterval     = 0.5;
};

class TrafficGymEnv : public OpenGymEnv {
public:
    static TypeId GetTypeId();
    TrafficGymEnv();
    TrafficGymEnv(SimConfig cfg);
    virtual ~TrafficGymEnv();

    Ptr<OpenGymSpace>         GetObservationSpace() override;
    Ptr<OpenGymSpace>         GetActionSpace()      override;
    Ptr<OpenGymDataContainer> GetObservation()      override;
    float                     GetReward()           override;
    bool                      GetGameOver()         override;
    std::string               GetExtraInfo()        override;
    bool ExecuteActions(Ptr<OpenGymDataContainer> action) override;

    void ScheduleNextStep();
    void Step();
    void CollectStats();
    void BuildLinearTopology();
    void BuildGridTopology();
    void BuildRandomTopology();
    void InstallStaticRoutes();
    void UpdateRoute(uint32_t nodeId, uint32_t nextHop);
    void InstallMultipleFlows(uint32_t numFlows);

private:
    SimConfig                m_cfg;
    NodeContainer            m_nodes;
    NetDeviceContainer       m_devices;
    Ipv4InterfaceContainer   m_interfaces;
    FlowMonitorHelper        m_flowHelper;
    Ptr<FlowMonitor>         m_flowMonitor;

    std::vector<Ptr<Ipv4StaticRouting>>       m_staticRouting;
    std::vector<std::pair<uint32_t,uint32_t>> m_links;

    // Per-node obs (5 metrics)
    std::vector<double> m_queueLen;
    std::vector<double> m_linkUtil;
    std::vector<double> m_delay;
    std::vector<double> m_lossRate;
    std::vector<double> m_congestion;

    // Global stats
    double   m_avgDelay;
    double   m_throughput;
    double   m_packetLoss;
    double   m_avgCongestion;
    uint32_t m_stepCount;

    // Previous step stats for delta reward
    double   m_prevDelay;
    double   m_prevTput;
    double   m_prevLoss;

    std::vector<std::string> m_bwLevels = {
        "1Mbps","5Mbps","10Mbps","50Mbps","100Mbps"
    };
};
