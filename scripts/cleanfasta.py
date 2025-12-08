from Bio import SeqIO
import os


def clean_fasta(input_fasta, output_fasta):
    cache_fasta = output_fasta + ".tmp"
    with open(cache_fasta, "w") as out_handle:
        for record in SeqIO.parse(input_fasta, "fasta"):
            seq_str = str(record.seq).replace("*", "")
            record.seq = record.seq.__class__(seq_str)

            # 只保留 id 作为序列名称
            record.description = record.id

            SeqIO.write(record, out_handle, "fasta")

    os.replace(cache_fasta, output_fasta)
