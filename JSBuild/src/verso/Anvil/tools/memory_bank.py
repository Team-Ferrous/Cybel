from pathlib import Path


def update_memory_bank(section: str, content: str, mode: str = "append") -> str:
    """
    Updates the GRANITE.md project memory bank.

    Args:
        section: Section name (e.g. 'Architecture', 'Conventions', 'Lessons Learned')
        content: The text to add or overwrite
        mode: 'append' to add to existing section, 'overwrite' to replace section content
    """
    path = Path("GRANITE.md")
    if not path.exists():
        # Create it if missing
        initial = "# Project Memory Bank\n\n## Architecture\n\n## Conventions\n\n## Lessons Learned\n"
        path.write_text(initial, encoding="utf-8")

    lines = path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    in_target_section = False
    section_found = False

    header = f"## {section}"

    for i, line in enumerate(lines):
        if line.strip().lower() == header.lower():
            new_lines.append(line)
            section_found = True
            in_target_section = True
            if mode == "overwrite":
                new_lines.append(content)
            else:  # append
                # We'll append at the end of this section
                pass
            continue

        if in_target_section and line.startswith("## "):
            # End of our section reached
            if mode == "append":
                new_lines.append(content)
            new_lines.append(line)
            in_target_section = False
            continue

        if not in_target_section:
            new_lines.append(line)

    if not section_found:
        # Add new section at the end
        if not new_lines[-1].strip() == "":
            new_lines.append("")
        new_lines.append(header)
        new_lines.append(content)
    elif in_target_section:
        # We were still in the section at the end of the file
        if mode == "append":
            new_lines.append(content)

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return f"Memory bank section '{section}' updated successfully."
