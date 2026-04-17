#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./package_agent.sh [random_agent|minmax_agent|mcts_agent|zero_agent|athenan_agent]"
    exit 1
fi

AGENT_NAME=$1
TARGET_ZIP="oxono.zip"
TEMP_FILE="my_agent.py"
FOLDER="files_99942500_26482510"

rm -f $TARGET_ZIP

case $AGENT_NAME in
    "random_agent")
        echo "Packaging random_agent..."
        cp random_agent.py $TEMP_FILE
        zip $TARGET_ZIP $TEMP_FILE
        ;;
    "minmax_agent")
        echo "Packaging minmax_agent..."
        cp minmax_agent.py $TEMP_FILE
        zip $TARGET_ZIP $TEMP_FILE "$FOLDER/zobrist.py" "$FOLDER/heuristic.py"
        ;;
    "mcts_agent")
        echo "Packaging mcts_agent..."
        cp mcts_agent.py $TEMP_FILE
        zip $TARGET_ZIP $TEMP_FILE "$FOLDER/heuristic.py" "$FOLDER/mcts.py" "$FOLDER/oxononet.py" "$FOLDER/utils.py"
        ;;
    "zero_agent")
        echo "Packaging zero_agent..."
        cp zero_agent.py $TEMP_FILE
        zip $TARGET_ZIP $TEMP_FILE "$FOLDER/config.py" "$FOLDER/mcts.py" "$FOLDER/oxononet.py" "$FOLDER/utils.py" "$FOLDER/zobrist.py" "$FOLDER/oxononet.pth"
        ;;
    "athenan_agent")
        echo "Packaging athenan_agent..."
        cp athenan_agent.py $TEMP_FILE
        zip $TARGET_ZIP $TEMP_FILE "$FOLDER/athenan_net.py" "$FOLDER/athenan_tree.py" "$FOLDER/utils.py" "$FOLDER/zobrist.py" "$FOLDER/athenan_oxono_model.pth"
        ;;
    *)
        echo "Error: Unknown agent name '$AGENT_NAME'. Please choose from [random_agent, minmax_agent, mcts_agent, zero_agent, athenan_agent]."
        exit 1
        ;;
esac

rm -f $TEMP_FILE
echo "Packaging complete: $TARGET_ZIP created successfully."
