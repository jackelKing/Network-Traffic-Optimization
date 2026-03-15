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

#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <map>
#include <utility>

using namespace ns3;

#define MAX_BW_LEVELS 5
#define OBS_PER_NODE  3

struct SimConfig {
    uint32_t    numNodes    = 4;
    std::string topoType    = "linear";
    double      simTime     = 10.0;
    uint16_t    openGymPort = 5555;
    std::string dataRate    = "10Mbps";
    std::string delay       = "2ms";
    double      delayWeight = 0.5;
    double      tputWeight  = 0.3;
    double      lossWeight  = 0.2;
    double      stepInterval = 0.5;
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

private:
    SimConfig                m_cfg;
    NodeContainer            m_nodes;
    NetDeviceContainer       m_devices;
    Ipv4InterfaceContainer   m_interfaces;
    FlowMonitorHelper        m_flowHelper;
    Ptr<FlowMonitor>         m_flowMonitor;

    // Static routing helpers per node
    std::vector<Ptr<Ipv4StaticRouting>> m_staticRouting;

    // Link map: (nodeA, nodeB) -> device index
    std::vector<std::pair<uint32_t,uint32_t>> m_links;

    std::vector<double>      m_queueLen;
    std::vector<double>      m_linkUtil;
    std::vector<double>      m_delay;

    double                   m_avgDelay;
    double                   m_throughput;
    double                   m_packetLoss;
    uint32_t                 m_stepCount;

    std::vector<std::string> m_bwLevels = {
        "1Mbps","5Mbps","10Mbps","50Mbps","100Mbps"
    };
};
