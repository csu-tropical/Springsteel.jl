#!/bin/bash
# Helper script for testing GitHub Actions locally with act
# Usage: ./test_ci.sh [command]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${2}${1}${NC}"
}

# Print header
print_header() {
    echo ""
    echo "========================================"
    print_msg "$1" "$BLUE"
    echo "========================================"
    echo ""
}

# Check if act is installed
check_act() {
    if ! command -v act &> /dev/null; then
        print_msg "❌ Error: 'act' is not installed" "$RED"
        echo ""
        echo "Install with:"
        echo "  macOS:   brew install act"
        echo "  Linux:   curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash"
        exit 1
    fi
    
    if ! docker ps &> /dev/null; then
        print_msg "❌ Error: Docker is not running" "$RED"
        echo ""
        echo "Please start Docker Desktop or the Docker daemon"
        exit 1
    fi
}

# Show help
show_help() {
    cat << EOF
🚀 Springsteel.jl CI Testing Script

Usage: ./test_ci.sh [command]

Commands:
  list          List all available workflows and jobs
  test          Run CI tests (recommended)
  test-quick    Run CI tests without matrix (faster)
  test-110      Run CI tests on Julia 1.10 only
  test-latest   Run CI tests on Julia latest only
  docs          Run documentation build
  dry-run       Show what would run without executing
  clean         Clean up Docker containers and images
  status        Show Docker container status
  help          Show this help message

Examples:
  ./test_ci.sh list         # See what's available
  ./test_ci.sh test         # Run all CI tests
  ./test_ci.sh test-110     # Test only Julia 1.10
  ./test_ci.sh docs         # Build documentation
  ./test_ci.sh dry-run      # Preview what will run

Integration tests are automatically enabled (SPARROW_RUN_INTEGRATION_TESTS=1)

For more details, see:
  .github/TESTING_WITH_ACT.md
  .github/ACT_QUICK_REFERENCE.md
EOF
}

# List workflows
list_workflows() {
    print_header "📋 Available Workflows and Jobs"
    act -l
}

# Run CI tests (full matrix)
run_ci_test() {
    print_header "🧪 Running CI Tests (Full Matrix)"
    print_msg "This will run tests on Julia 1.10 and Julia latest" "$YELLOW"
    print_msg "Integration tests are ENABLED" "$GREEN"
    echo ""
    act -W .github/workflows/CI.yml
}

# Run CI tests without matrix (faster)
run_ci_test_quick() {
    print_header "⚡ Running CI Tests (Quick - Julia latest only)"
    print_msg "Integration tests are ENABLED" "$GREEN"
    echo ""
    act -W .github/workflows/CI.yml --matrix version:1
}

# Run CI tests on Julia 1.10
run_ci_test_110() {
    print_header "🧪 Running CI Tests (Julia 1.10 only)"
    print_msg "Integration tests are ENABLED" "$GREEN"
    echo ""
    act -W .github/workflows/CI.yml --matrix version:1.10
}

# Run CI tests on Julia latest
run_ci_test_latest() {
    print_header "🧪 Running CI Tests (Julia latest only)"
    print_msg "Integration tests are ENABLED" "$GREEN"
    echo ""
    act -W .github/workflows/CI.yml --matrix version:1
}

# Run documentation build
run_docs() {
    print_header "📚 Running Documentation Build"
    print_msg "Note: Deployment will fail (requires GitHub credentials)" "$YELLOW"
    print_msg "But the build step should work fine" "$YELLOW"
    echo ""
    act -W .github/workflows/documentation.yml
}

# Dry run
run_dry_run() {
    print_header "🔍 Dry Run - CI Workflow"
    print_msg "Showing what would run without executing..." "$YELLOW"
    echo ""
    act -W .github/workflows/CI.yml -n | head -100
    echo ""
    print_msg "..." "$YELLOW"
    print_msg "(Output truncated - run 'act -W .github/workflows/CI.yml -n' for full output)" "$YELLOW"
}

# Clean up
clean_up() {
    print_header "🧹 Cleaning Up Docker Resources"
    
    echo "Removing act containers..."
    docker ps -a | grep act- | awk '{print $1}' | xargs -r docker rm -f || true
    
    echo "Pruning Docker system..."
    docker system prune -f
    
    print_msg "✅ Cleanup complete" "$GREEN"
}

# Show Docker status
show_status() {
    print_header "📊 Docker Container Status"
    echo "Act containers currently running:"
    docker ps | grep act- || print_msg "No act containers running" "$YELLOW"
    echo ""
    echo "Docker disk usage:"
    docker system df
}

# Main script
main() {
    check_act
    
    case "${1:-help}" in
        list)
            list_workflows
            ;;
        test)
            run_ci_test
            ;;
        test-quick|quick)
            run_ci_test_quick
            ;;
        test-110|110)
            run_ci_test_110
            ;;
        test-latest|latest)
            run_ci_test_latest
            ;;
        docs|documentation)
            run_docs
            ;;
        dry-run|dryrun|preview)
            run_dry_run
            ;;
        clean|cleanup)
            clean_up
            ;;
        status)
            show_status
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_msg "❌ Unknown command: $1" "$RED"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
