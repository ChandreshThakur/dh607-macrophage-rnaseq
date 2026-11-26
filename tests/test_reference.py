import gzip

from fastq_pipeline.reference import fasta_tx2gene, gtf_tx2gene, parse_gtf_attributes


def test_parse_gtf_attributes():
    result = parse_gtf_attributes('gene_id "ENSG1"; transcript_id "ENST1"; gene_name "X";')
    assert result == {"gene_id": "ENSG1", "transcript_id": "ENST1", "gene_name": "X"}


def test_gtf_tx2gene(tmp_path):
    path = tmp_path / "tiny.gtf.gz"
    with gzip.open(path, "wt") as handle:
        handle.write('21\tens\ttranscript\t1\t10\t.\t+\t.\tgene_id "ENSG1"; transcript_id "ENST1";\n')
        handle.write('21\tens\texon\t1\t10\t.\t+\t.\tgene_id "ENSG1"; transcript_id "ENST1";\n')
    result = gtf_tx2gene(path)
    assert result.to_dict("records") == [{"transcript_id": "ENST1", "gene_id": "ENSG1"}]


def test_fasta_tx2gene_removes_gene_version(tmp_path):
    path = tmp_path / "tiny.fa"
    path.write_text(">ENST1.2 cdna gene:ENSG1.7 gene_biotype:protein_coding\nACGT\n")
    result = fasta_tx2gene(path)
    assert result.to_dict("records") == [{"transcript_id": "ENST1.2", "gene_id": "ENSG1"}]
