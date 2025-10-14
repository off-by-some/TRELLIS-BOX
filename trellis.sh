#!/bin/bash

# TRELLIS Docker Management Script
# Main entry point that dispatches to individual scripts

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

print_usage() {
    echo "🚀 TRELLIS Docker Manager"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  run      - Start TRELLIS (builds image, checks GPU, etc.)"
    echo "  restart  - Restart a stopped TRELLIS container"
    echo "  stop     - Stop the running TRELLIS container"
    echo "  status   - Show TRELLIS status and system info"
    echo "  build    - Build the Docker image only"
    echo "  check    - Check GPU memory availability"
    echo ""
    echo "Examples:"
    echo "  $0 run     # Start TRELLIS with full setup"
    echo "  $0 status  # Check current status"
    echo "  $0 stop    # Stop TRELLIS"
}

# Check if scripts directory exists
if [ ! -d "scripts" ]; then
    print_error "scripts/ directory not found. Are you in the right directory?"
    exit 1
fi

# Get the command
COMMAND="$1"

# Validate command
case "$COMMAND" in
    run|restart|stop|status|build|check)
        SCRIPT="scripts/${COMMAND}.sh"
        if [ ! -f "$SCRIPT" ]; then
            print_error "Script '$SCRIPT' not found"
            exit 1
        fi

        # Execute the script with any additional arguments
        shift  # Remove the command from arguments
        exec "$SCRIPT" "$@"
        ;;
    "")
        print_usage
        exit 1
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        echo ""
        print_usage
        exit 1
        ;;
esac
