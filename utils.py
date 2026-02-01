def fetch_sequences_from_fasta(sequence_fpath):
    from Bio import SeqIO
    sequence_names = []
    sequence_list = []
    sequence_descriptions = []
    for j, record in enumerate(SeqIO.parse(sequence_fpath, "fasta")):
        sequence_names.append(record.id)
        sequence_list.append(str(record.seq))
        sequence_descriptions.append(record.description)
    return sequence_list, sequence_names, sequence_descriptions