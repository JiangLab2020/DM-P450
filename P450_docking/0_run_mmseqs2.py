#!/usr/bin/env python3
import os
import sys
import time
import subprocess
from pathlib import Path
from multiprocessing import cpu_count


def run_cmd(cmd):
    """Run shell commands safely."""
    subprocess.run(cmd, shell=True, check=True)


def clean_and_split_fasta(input_fasta, clean_fasta_dir):
    """Clean sequences (A-Z only) and split into individual FASTA files."""
    os.makedirs(clean_fasta_dir, exist_ok=True)
    seq = ""
    header = None

    with open(input_fasta, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith(">"):
                if header is not None and seq != "":
                    seq = "".join([c for c in seq if c.isalpha() and c.isupper()])
                    out_file = os.path.join(clean_fasta_dir, f"{header}.fasta")
                    with open(out_file, "w") as out:
                        out.write(f">{header}\n{seq}\n")

                header = line[1:].split()[0]
                seq = ""
            else:
                seq += line

    # save last sequence
    if header is not None and seq != "":
        seq = "".join([c for c in seq if c.isalpha() and c.isupper()])
        out_file = os.path.join(clean_fasta_dir, f"{header}.fasta")
        with open(out_file, "w") as out:
            out.write(f">{header}\n{seq}\n")

    return sorted(Path(clean_fasta_dir).glob("*.fasta"))


def print_progress(count, total):
    """ASCII progress bar."""
    percent = (count / total) * 100
    bar_len = 50
    filled = int(percent / 100 * bar_len)
    bar = "█" * filled + " " * (bar_len - filled)
    print(f"\rProgress: [{bar}] {percent:.2f}% ({count}/{total})", end="")


def clean_a3m(input_path, output_path):
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYBJZUX*-")  # 合法氨基酸字符
    
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            line = line.rstrip('\n')
            if line.startswith('>'):
                fout.write(line + '\n')
            else:
                cleaned_seq = ''.join([c for c in line if c.upper() in valid_aa])
                if cleaned_seq:
                    fout.write(cleaned_seq + '\n')

def batch_clean(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".a3m"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f"处理文件: {filename}")
            clean_a3m(input_path, output_path)
    print("全部处理完成。")


def main():
    input_fasta = sys.argv[1]

    if not os.path.isfile(input_fasta):
        print(f"错误：找不到输入文件 {input_fasta}")
        sys.exit(1)

    target_db = "/home/hanxudong/liushuo/mmseq2_test/uniref50/uniref50_db"

    if len(sys.argv) > 2:
        target_db = sys.argv[2]

    start_time = time.time()

    # === 修复 dirname("") → "." 的问题 ===
    raw_workdir = os.path.dirname(input_fasta)
    if not raw_workdir:
        raw_workdir = "."
    workdir = os.path.abspath(raw_workdir)

    basename = os.path.splitext(os.path.basename(input_fasta))[0]

    db_dir = f"{workdir}/db"
    tmp_dir = f"{workdir}/tmp_dir"
    result_m8_dir = f"{workdir}/result_m8"
    msa_output_dir = f"{workdir}/msa_output"
    clean_fasta_dir = f"{workdir}/clean_fasta"

    Target = input_fasta.split('/')[-1].split('.')[0]
    msa_output_dir_clean = f"{workdir}/{Target}-msa_cleaned"

    threads = cpu_count()


    # create directories
    for d in [db_dir, tmp_dir, result_m8_dir, msa_output_dir, clean_fasta_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"Step 1: Cleaning and splitting sequences from {input_fasta}...")
    fasta_files = clean_and_split_fasta(input_fasta, clean_fasta_dir)
    total = len(fasta_files)
    count = 0

    print("Step 2: Running MMseqs2...")

    for fasta_file in fasta_files:
        count += 1
        seq_name = fasta_file.stem

        print(f"\nGene process: {count}")
        print_progress(count, total)

        query_db = f"{db_dir}/{seq_name}_DB"
        result_dir = f"{workdir}/result_dir_{seq_name}"
        result_m8 = f"{result_m8_dir}/{seq_name}.m8"
        msa_output = f"{msa_output_dir}/{seq_name}.a3m"

        run_cmd(f"mmseqs createdb {fasta_file} {query_db}")
        run_cmd(f"mmseqs search {query_db} {target_db} {result_dir} {tmp_dir} --threads {threads}")
        run_cmd(f"mmseqs convertalis {query_db} {target_db} {result_dir} {result_m8}")
        run_cmd(f"mmseqs result2msa {query_db} {target_db} {result_dir} {msa_output} --msa-format-mode 5 --threads {threads}")

    print_progress(total, total)
    print("\n\n✅ 所有序列处理完成！")
    print(f"    - 比对结果 (.m8) 保存在: {result_m8_dir}")
    print(f"    - MSA 文件 (.a3m) 保存在: {msa_output_dir}")
    print(f"    - 清理后的 fasta 文件保存在: {clean_fasta_dir}")

    print("Step 3: Cleaning up intermediate files...")
    run_cmd(f"rm -rf {db_dir} {tmp_dir}")
    run_cmd(f"rm -rf {workdir}/result_dir_*")
    run_cmd(f"rm -f {workdir}/*.dbtype {workdir}/*.index")

    runtime = int(time.time() - start_time)
    runtime_min = runtime / 60
    print(f"⏱️ 总运行时间: {runtime} 秒（约 {runtime_min:.2f} 分钟）")

    batch_clean(msa_output_dir, msa_output_dir_clean)

    os.system('rm -f result_dir_*')
    os.system('rm -f temp_*')

    os.system('rm -rf db')
    os.system('rm -rf msa_output')
    os.system('rm -rf result_m8')
    os.system('rm -rf tmp_dir')

    os.system('rm -rf clean_fasta')


if __name__ == "__main__":
    main()
