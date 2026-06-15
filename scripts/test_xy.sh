#!/bin/bash
################################### test ######################################

CYCLE=100000
LINK_WIDTH=128
LINK_LATENCY=1
ROUTER_LATENCY=1
VNET=1
TOPOLOGY=Mesh_XY
ALGORITHM=Mesh_XY

if [[ "$ALGORITHM" == "Table" ]]; then
    ALGORITHM_CODE=0
elif [[ "$ALGORITHM" == "Mesh_XY" ]]; then
    ALGORITHM_CODE=1
elif [[ "$ALGORITHM" == "Odd_Even" ]]; then
    ALGORITHM_CODE=2
elif [[ "$ALGORITHM" == "Negative_First" ]]; then
    ALGORITHM_CODE=3
elif [[ "$ALGORITHM" == "Aware" ]]; then
    ALGORITHM_CODE=4
elif [[ "$ALGORITHM" == "Fault" ]]; then
    ALGORITHM_CODE=5
fi

mkdir -p temp_latency
CSV_LATENCY=temp_latency/$ALGORITHM
touch $CSV_LATENCY

mkdir -p job
FILE=$ALGORITHM.dat
touch $FILE

echo "algorithm is $ALGORITHM" > $FILE
echo "algorithm is $ALGORITHM" > $CSV_LATENCY

TICK_TO_CYCLE=1000

for MODE in uniform_random
do
for INJECTIONRATE in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do

    build/Garnet_standalone/gem5.opt configs/example/garnet_synth_traffic.py \
        --num-cpus=64 \
        --num-dirs=64 \
        --mesh-rows=8 \
        --network=garnet \
        --topology=$TOPOLOGY \
        --link-latency=$LINK_LATENCY \
        --router-latency=$ROUTER_LATENCY \
        --inj-vnet=$VNET \
        --sim-cycles=$CYCLE \
        --synthetic=$MODE \
        --injectionrate=$INJECTIONRATE \
        --routing-algorithm=$ALGORITHM_CODE \
        --link-width-bits=$LINK_WIDTH \
        > test.txt

    mkdir -p ./job/$ALGORITHM/$MODE/$INJECTIONRATE

    FLIT_LAT=$(grep -w "system.ruby.network.average_flit_latency" m5out/stats.txt 2>/dev/null | awk '{print $2}')
    PACKET_LAT=$(grep -w "system.ruby.network.average_packet_latency" m5out/stats.txt 2>/dev/null | awk '{print $2}')
    FLIT_NET_LAT=$(grep -w "system.ruby.network.flit_network_latency" m5out/stats.txt 2>/dev/null | awk '{print $2}')
    FLITS_RECV=$(grep -w "system.ruby.network.flits_received::total" m5out/stats.txt 2>/dev/null | awk '{print $2}')
    HOPS=$(grep -w "system.ruby.network.average_hops" m5out/stats.txt 2>/dev/null | awk '{print $2}')

    # Safe division: only compute if value is non-empty
    if [ -n "$FLIT_LAT" ]; then
        FLIT_LAT_CYCLE=$(awk "BEGIN {printf \"%.4f\", $FLIT_LAT / $TICK_TO_CYCLE}")
    else
        FLIT_LAT_CYCLE="N/A"
    fi
    if [ -n "$PACKET_LAT" ]; then
        PACKET_LAT_CYCLE=$(awk "BEGIN {printf \"%.4f\", $PACKET_LAT / $TICK_TO_CYCLE}")
    else
        PACKET_LAT_CYCLE="N/A"
    fi
    if [ -n "$FLIT_NET_LAT" ]; then
        FLIT_NET_LAT_CYCLE=$(awk "BEGIN {printf \"%.4f\", $FLIT_NET_LAT / $TICK_TO_CYCLE}")
    else
        FLIT_NET_LAT_CYCLE="N/A"
    fi

    FLITS_RECV=${FLITS_RECV:-"N/A"}
    HOPS=${HOPS:-"N/A"}

    echo "" >> $FILE
    echo "sim-cycles is $CYCLE" >> $FILE
    echo "mode is $MODE" >> $FILE
    echo "injectionrate is $INJECTIONRATE" >> $FILE
    echo "average_flit_latency: $FLIT_LAT_CYCLE" >> $FILE
    echo "average_hops: $HOPS" >> $FILE

    cp -r m5out ./job/$ALGORITHM/$MODE/$INJECTIONRATE
    cp test.txt ./job/$ALGORITHM/$MODE/$INJECTIONRATE

    echo "$INJECTIONRATE,$FLIT_LAT_CYCLE" >> $CSV_LATENCY

done
done
cp $FILE ./job/$ALGORITHM
cp $CSV_LATENCY ./job/$ALGORITHM
