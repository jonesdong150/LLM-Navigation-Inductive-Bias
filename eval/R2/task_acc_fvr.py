import os
import json

# Define root directory containing all model result directories
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis_ieq_dim")
# Define output TXT file path
OUTPUT_TXT = os.path.join(ROOT_DIR, "task_acc_fvr.txt")
# Target structures and tasks from JSON example
TARGET_STRUCTURES = ["clustered", "clustered_25", "clustered_50", "flat", "hier", "hier_25", "hier_50"]
TARGET_TASKS = ["ObjectLocation", "GeometryYN", "TopologyYN", "ReachabilityYN", "PathGen"]

# Define table column widths (for alignment)
COL_WIDTHS = {
    "model": 20,
    "scene": 10,
    "structure": 15,
    "task": 20,
    "accuracy": 15,
    "fvr": 25
}

def format_cell(content, width, align="left"):
    """Format cell content with alignment"""
    content_str = str(content)
    if align == "left":
        return content_str.ljust(width)[:width]
    elif align == "right":
        return content_str.rjust(width)[:width]
    elif align == "center":
        return content_str.center(width)[:width]

def main():
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f_out:
        # Write table title
        f_out.write("Model Inference Results Statistics (accuracy & format_violation_rate)\n")
        f_out.write("=" * (sum(COL_WIDTHS.values()) + 5) + "\n\n")

        # Write table header
        header = (
            format_cell("Model Name", COL_WIDTHS["model"]) + "|" +
            format_cell("Scene", COL_WIDTHS["scene"], "center") + "|" +
            format_cell("Structure", COL_WIDTHS["structure"], "center") + "|" +
            format_cell("Task", COL_WIDTHS["task"]) + "|" +
            format_cell("accuracy", COL_WIDTHS["accuracy"], "center") + "|" +
            format_cell("format_violation_rate", COL_WIDTHS["fvr"], "center")
        )
        f_out.write(header + "\n")
        # Write header separator
        separator = (
            "-" * COL_WIDTHS["model"] + "+" +
            "-" * COL_WIDTHS["scene"] + "+" +
            "-" * COL_WIDTHS["structure"] + "+" +
            "-" * COL_WIDTHS["task"] + "+" +
            "-" * COL_WIDTHS["accuracy"] + "+" +
            "-" * COL_WIDTHS["fvr"]
        )
        f_out.write(separator + "\n")

        # Traverse all model directories
        for item in os.listdir(ROOT_DIR):
            model_dir = os.path.join(ROOT_DIR, item)
            if not os.path.isdir(model_dir):
                continue  # Filter non-directory files (e.g., R2_result_all.txt)

            # Traverse 10 scenes
            for scene_idx in range(1, 11):
                scene_file_name = f"scene_complex_{scene_idx:02d}_summary.json"
                scene_file_path = os.path.join(model_dir, scene_file_name)

                # Handle missing file
                if not os.path.exists(scene_file_path):
                    row = (
                        format_cell(item, COL_WIDTHS["model"]) + "|" +
                        format_cell(f"{scene_idx:02d}", COL_WIDTHS["scene"], "center") + "|" +
                        format_cell("--", COL_WIDTHS["structure"], "center") + "|" +
                        format_cell("File not found", COL_WIDTHS["task"]) + "|" +
                        format_cell("--", COL_WIDTHS["accuracy"], "center") + "|" +
                        format_cell("--", COL_WIDTHS["fvr"], "center")
                    )
                    f_out.write(row + "\n")
                    continue

                # Read and parse JSON file
                try:
                    with open(scene_file_path, "r", encoding="utf-8") as f_json:
                        scene_data = json.load(f_json)
                except json.JSONDecodeError:
                    row = (
                        format_cell(item, COL_WIDTHS["model"]) + "|" +
                        format_cell(f"{scene_idx:02d}", COL_WIDTHS["scene"], "center") + "|" +
                        format_cell("--", COL_WIDTHS["structure"], "center") + "|" +
                        format_cell("JSON parse failed", COL_WIDTHS["task"]) + "|" +
                        format_cell("--", COL_WIDTHS["accuracy"], "center") + "|" +
                        format_cell("--", COL_WIDTHS["fvr"], "center")
                    )
                    f_out.write(row + "\n")
                    continue
                except Exception as e:
                    row = (
                        format_cell(item, COL_WIDTHS["model"]) + "|" +
                        format_cell(f"{scene_idx:02d}", COL_WIDTHS["scene"], "center") + "|" +
                        format_cell("--", COL_WIDTHS["structure"], "center") + "|" +
                        format_cell(f"Read error: {str(e)}", COL_WIDTHS["task"]) + "|" +
                        format_cell("--", COL_WIDTHS["accuracy"], "center") + "|" +
                        format_cell("--", COL_WIDTHS["fvr"], "center")
                    )
                    f_out.write(row + "\n")
                    continue

                # Traverse all structures
                for structure in TARGET_STRUCTURES:
                    if structure not in scene_data:
                        row = (
                            format_cell(item, COL_WIDTHS["model"]) + "|" +
                            format_cell(f"{scene_idx:02d}", COL_WIDTHS["scene"], "center") + "|" +
                            format_cell(structure, COL_WIDTHS["structure"], "center") + "|" +
                            format_cell("No structure data", COL_WIDTHS["task"]) + "|" +
                            format_cell("--", COL_WIDTHS["accuracy"], "center") + "|" +
                            format_cell("--", COL_WIDTHS["fvr"], "center")
                        )
                        f_out.write(row + "\n")
                        continue

                    struct_data = scene_data[structure]
                    # Traverse all tasks
                    for task in TARGET_TASKS:
                        if task not in struct_data["tasks"]:
                            row = (
                                format_cell(item, COL_WIDTHS["model"]) + "|" +
                                format_cell(f"{scene_idx:02d}", COL_WIDTHS["scene"], "center") + "|" +
                                format_cell(structure, COL_WIDTHS["structure"], "center") + "|" +
                                format_cell(task, COL_WIDTHS["task"]) + "|" +
                                format_cell("No task data", COL_WIDTHS["accuracy"], "center") + "|" +
                                format_cell("No task data", COL_WIDTHS["fvr"], "center")
                            )
                            f_out.write(row + "\n")
                            continue

                        # Extract valid data
                        task_data = struct_data["tasks"][task]
                        accuracy = task_data.get("accuracy", "No data")
                        fvr = task_data.get("format_violation_rate", "No data")

                        # Build table row
                        row = (
                            format_cell(item, COL_WIDTHS["model"]) + "|" +
                            format_cell(f"{scene_idx:02d}", COL_WIDTHS["scene"], "center") + "|" +
                            format_cell(structure, COL_WIDTHS["structure"], "center") + "|" +
                            format_cell(task, COL_WIDTHS["task"]) + "|" +
                            format_cell(accuracy, COL_WIDTHS["accuracy"], "center") + "|" +
                            format_cell(fvr, COL_WIDTHS["fvr"], "center")
                        )
                        f_out.write(row + "\n")

            # Add separator between models
            f_out.write(separator + "\n\n")

    print(f"Tabular statistics saved to: {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
