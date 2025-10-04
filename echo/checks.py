from pathlib import Path
import sys

def req_files() :
    print("Checking if all files are there.\n")
    crit_error_count = 0

    if getattr(sys, 'frozen', False):
        base_path = Path(sys.executable).parent
    else:
        base_path = Path(__file__).resolve().parent.parent

    file_path = base_path / "data"
    print(file_path)
    print()

    if file_path.exists():
        print(f"The path '{file_path}' exists.")
        chrom_path = file_path / "chrom_sorted_cpg_probes"
        if chrom_path.exists():
            print(f"The path '{chrom_path}' exists.")
            for i in range(1, 25) :
                if i <= 22 :
                    file_name = "chr"+str(i)+".csv"
                elif i == 23 :
                    file_name = "chrX.csv"
                elif i == 24 :
                    file_name = "chrY.csv"
                csv_path = chrom_path / file_name
                if not csv_path.is_file():
                    print(f"CRITICAL ERROR : The file '{file_name}' does not exist!")
                    crit_error_count += 1
                    break
            print("All chrom-sorted cpg probe files exist.")

            csv_path = file_path / "geneEx_data.csv"
            if csv_path.is_file():
                print(f"'{csv_path}' found.")
            else:
                print(f"CRITICAL ERROR : The file '{csv_path}' does not exist!")
                crit_error_count += 1

            map_path = file_path / "probeMap_hugo_gencode_good_hg19_V24lift37_probemap"
            if map_path.is_file() :
                print(f"'{map_path}' found.")
            else:
                print(f"CRITICAL ERROR : The file '{map_path}' does not exist!")
                crit_error_count += 1

            map_path = file_path / "probeMap_illuminaMethyl450_hg19_GPL16304_TCGAlegacy"
            if map_path.is_file() :
                print(f"'{map_path}' found.")
            else:
                print(f"CRITICAL ERROR : The file '{map_path}' does not exist!")
                crit_error_count += 1
        else:
            print(f"CRITICAL ERROR : The path '{chrom_path}' does not exist!")
            crit_error_count += 1
    else:
        print(f"CRITICAL ERROR : The path '{file_path}' does not exist!")
        crit_error_count += 1

    print()
    print(f"Initialization finished with {crit_error_count} error(s)!")