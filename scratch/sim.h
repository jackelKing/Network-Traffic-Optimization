#pragma once

#include "ns3/opengym-module.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/traffic-control-module.h"

#include <vector>
#include <map>
#include <string>

using namespace ns3;

// ─── Constants ───────────────────────────────────────────────
#define MAX_NODES     20
#define MAX_BW_LEVELS  5    // discrete bandwidth levels per link
#define OBS_PER_NODE   3    // queue_len, link_util, delay per node

// ─── Global simulation parameters ────────────────────────────
struct SimConfig {
    uint32_t    numNodes    = 4;
    std::string topoType    = "linear";   // linear | grid | random
    double      simTime     = 10.0;
    uint16_t    openGymPort = 5555;
    std::string dataRate    = "10Mbps";
    std::string delay       = "2ms";
    double      delayWeight = 0.5;
    double      tputWeight  = 0.3;
    double      lossWeight  = 0.2;
};

// ─── OpenGym environment class ────────────────────────────────
class TrafficGymEnv : public OpenGymEnv {
public:
    static TypeId GetTypeId();
    TrafficGymEnv(SimConfig cfg);
    virtual ~TrafficGymEnv();

    // Required OpenGymEnv overrides
    Ptr<OpenGymSpace>   GetObservationSpace() override;
    Ptr<OpenGymSpace>   GetActionSpace()      override;
    Ptr<OpenGymDataContainer> GetObservation() override;
    float               GetReward()           override;
    bool                GetGameOver()         override;
    std::string         GetExtraInfo()        override;
    bool                ExecuteActions(Ptr<OpenGymDataContainer> action) override;

    // Called every step to update internal stats
    void CollectStats();

    // Topology builders
    void BuildLinearTopology();
    void BuildGridTopology();
    void BuildRandomTopology();

private:
    SimConfig                        m_cfg;
    NodeContainer                    m_nodes;
    NetDeviceContainer               m_devices;
    Ipv4InterfaceContainer           m_interfaces;
    FlowMonitorHelper                m_flowHelper;
    Ptr<FlowMonitor>                 m_flowMonitor;

    // Per-node stats (updated each step)
    std::vector<double>              m_queueLen;
    std::vector<double>              m_linkUtil;
    std::vector<double>              m_delay;

    // Reward components (updated each step)
    double                           m_avgDelay;
    double                           m_throughput;
    double                           m_packetLoss;

    uint32_t                         m_stepCount;
    bool                             m_done;

    // Bandwidth level → string mapping
    std::vector<std::string>         m_bwLevels = {
        "1Mbps", "5Mbps", "10Mbps", "50Mbps", "100Mbps"
    };
};
