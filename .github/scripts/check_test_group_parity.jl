#!/usr/bin/env julia
#
# Sanity check: ensure test/runtests.jl and .github/workflows/CI.yml list
# the same set of test groups. Runs in CI before the matrix job and is
# also runnable locally from the repo root:
#
#     julia .github/scripts/check_test_group_parity.jl
#
# Exits 0 if the two files agree, 1 with a clear diff otherwise. No
# external dependencies — pure Base string matching.

const REPO_ROOT      = normpath(joinpath(@__DIR__, "..", ".."))
const RUNTESTS_PATH  = joinpath(REPO_ROOT, "test", "runtests.jl")
const CI_YAML_PATH   = joinpath(REPO_ROOT, ".github", "workflows", "CI.yml")

"""
    parse_runtests_groups(path::AbstractString) -> Set{String}

Extract every test group name from a `test/runtests.jl` that follows the
`TEST_GROUP in ("all", "<name>") && include("...")` convention. The
literal `"all"` is excluded — we only collect real group names.
"""
function parse_runtests_groups(path::AbstractString)
    src = read(path, String)
    groups = Set{String}()
    # Captures the <name> in: TEST_GROUP in ("all", "<name>")
    pat = r"TEST_GROUP\s+in\s+\(\s*\"all\"\s*,\s*\"([A-Za-z0-9_]+)\"\s*\)"
    for m in eachmatch(pat, src)
        push!(groups, m.captures[1])
    end
    return groups
end

"""
    parse_ci_groups(path::AbstractString) -> Set{String}

Extract every matrix group name from `.github/workflows/CI.yml`. Walks
the file line by line and collects list entries that sit under the
`group:` key of a matrix — i.e. lines matching `- <name>` at an
indentation strictly deeper than the `group:` line itself.
"""
function parse_ci_groups(path::AbstractString)
    groups = Set{String}()
    in_group_list      = false
    group_line_indent  = -1
    for raw in eachline(path)
        # Strip comments so `# group: foo` doesn't trigger the state machine.
        line = replace(raw, r"#.*$" => "")
        isempty(strip(line)) && continue

        # Leading whitespace count (spaces only — YAML forbids tabs).
        m_indent = match(r"^(\s*)", line)
        indent = m_indent === nothing ? 0 : length(m_indent.captures[1])

        if occursin(r"^\s*group:\s*$", line)
            in_group_list = true
            group_line_indent = indent
            continue
        end

        if in_group_list
            # A list item deeper than the group: line is a group name.
            m = match(r"^(\s*)-\s+([A-Za-z0-9_]+)\s*$", line)
            if m !== nothing && length(m.captures[1]) > group_line_indent
                push!(groups, m.captures[2])
            elseif indent <= group_line_indent
                # De-dented back out of the group list — stop collecting.
                in_group_list = false
            end
        end
    end
    return groups
end

function main()
    rt_groups = parse_runtests_groups(RUNTESTS_PATH)
    ci_groups = parse_ci_groups(CI_YAML_PATH)

    if isempty(rt_groups)
        println(stderr, "ERROR: parser found zero groups in $RUNTESTS_PATH — ",
                "has the TEST_GROUP in (\"all\", \"...\") convention changed? ",
                "This script needs an update.")
        exit(2)
    end
    if isempty(ci_groups)
        println(stderr, "ERROR: parser found zero groups in $CI_YAML_PATH — ",
                "has the matrix structure changed? This script needs an update.")
        exit(2)
    end

    missing_from_ci = sort!(collect(setdiff(rt_groups, ci_groups)))
    missing_from_rt = sort!(collect(setdiff(ci_groups, rt_groups)))

    if isempty(missing_from_ci) && isempty(missing_from_rt)
        println("Test group parity OK — $(length(rt_groups)) groups match between ",
                "test/runtests.jl and .github/workflows/CI.yml.")
        return 0
    end

    println(stderr, "Test group parity MISMATCH between test/runtests.jl and ",
                    ".github/workflows/CI.yml")
    if !isempty(missing_from_ci)
        println(stderr)
        println(stderr, "  Groups in runtests.jl but missing from CI matrix:")
        for g in missing_from_ci
            println(stderr, "    - $g")
        end
        println(stderr, "  → Add these to the `group:` list in .github/workflows/CI.yml")
    end
    if !isempty(missing_from_rt)
        println(stderr)
        println(stderr, "  Groups in CI matrix but not wired into runtests.jl:")
        for g in missing_from_rt
            println(stderr, "    - $g")
        end
        println(stderr, "  → Add `TEST_GROUP in (\"all\", \"<name>\") && include(...)` ",
                        "entries to test/runtests.jl")
    end
    return 1
end

exit(main())
