#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./package_agent.sh [minmax_agent|zero_agent]"
    exit 1
fi

AGENT_NAME=$1
TARGET_ZIP="oxono.zip"
TEMP_FILE="my_agent.py"
FOLDER="files_99942500_26482510"

rm -f $TARGET_ZIP

case $AGENT_NAME in
    "minmax_agent")
        echo "Packaging de l'agent MinMax..."
        cp minmax_agent.py $TEMP_FILE
        zip $TARGET_ZIP $TEMP_FILE "$FOLDER/zobrist.py"
        ;;
    "zero_agent")
        echo "Packaging de l'agent Zero..."
        cp zero_agent.py $TEMP_FILE
        zip $TARGET_ZIP $TEMP_FILE "$FOLDER"/config.py "$FOLDER"/mcts.py "$FOLDER"/oxononet.py "$FOLDER"/utils.py "$FOLDER"/zobrist.py "$FOLDER/oxononet.pth"
        ;;
    *)
        echo "❌ Erreur : Seuls 'minmax_agent' et 'zero_agent' sont acceptés."
        exit 1
        ;;
esac

rm -f $TEMP_FILE
echo "Terminé : $TARGET_ZIP a été généré avec succès."
