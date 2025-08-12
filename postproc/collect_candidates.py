import os
import glob

def collect_candidates(sap_dir, output_filename="combined_all_candidates.cands"):
    output_path = os.path.join(sap_dir, output_filename)
    beam_dirs = sorted(glob.glob(os.path.join(sap_dir, "B???")))

    print(f"Found {len(beam_dirs)} beam directories.")

    all_lines = []
    for beam_dir in beam_dirs:
        beam_name = os.path.basename(beam_dir)
        cands_file = os.path.join(beam_dir, "output", "all_detected_candidates.cands")
        
        if os.path.isfile(cands_file):
            with open(cands_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        all_lines.append(f"{beam_name} {line}")
            print(f"  - {beam_name}: added candidates from {cands_file}")
        else:
            print(f"  - {beam_name}: No candidate file found at expected path: {cands_file}")

    if all_lines:
        header = "# Beam  DM(pc/cm^3)  Detection Strength  Time(s)  Sample  Filter Width(samples)"
        with open(output_path, "w") as out:
            out.write(header + "\n")
            out.write("\n".join(all_lines) + "\n")
        print(f"\n✅ Combined candidate list saved to: {output_path}")
    else:
        print("\n⚠️ No candidate lines found to combine.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combine candidate files for a single SAP, including beam info.")
    parser.add_argument("sap_dir", help="Path to SAP directory (e.g., /path/to/SAP001)")
    args = parser.parse_args()

    collect_candidates(args.sap_dir)