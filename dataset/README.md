# Proteoformer Corpus: A Comprehensive Proteoform Dataset for Foundation Model Training

## Overview

The `Proteoformer Corpus` represents the largest curated collection of proteoforms, comprising over `1.9 million` distinct forms, developed to support the training of proteoform foundation models. This comprehensive dataset integrates and harmonizes proteoform data from multiple authoritative sources, providing an extensive resource for studying protein diversity and developing advanced computational methods for proteoform analysis.

## Background

Proteoforms represent the different molecular forms in which a protein can exist, arising from genetic variations, alternative splicing, and post-translational modifications (PTMs). The Consortium for Top Down Proteomics (CTDP) developed the ProForma notation format to standardize proteoform representation. However, to accommodate diverse research communities and large-scale model training requirements, we have developed the PFMOD encoding scheme, which provides a unified representation for amino acids and modification tokens.

## Data Sources and Integration

This corpus integrates proteoform data from three major sources:

1. **UniProtKB/Swiss-Prot (May 2024)**: Provides canonical protein sequences and annotations from the manually reviewed Swiss-Prot database
2. **Human Proteoform Atlas (HPfA) (May 2024)**: Offers experimentally validated proteoforms from the comprehensive human proteoform repository
3. **dbPTM V2**: Contains comprehensive post-translational modification data from the updated dbPTM database

## Dataset Statistics

| Dataset Name | Source | Form counts | Sequences | Token counts | Avg Length (token) |
|--------------|-------|-------------|------------|------------|-----------------|
| dbPTM corpus | dbPTM | 1,731,194 | 287,075 | 174,547,310 | 608 |
| Variant corpus | UniprotKB/Swiss-prot | 102,653 | 102,653 | 106,917,926 | 1041 |
| HPfA corpus | Human Proteoform Atlas | 61,770 | 61,770 | 7,753,932 | 125 |
| Isoform corpus | UniprotKB/Swiss-prot | 52,927 | 52,927 | 37,390,292 | 706 |
| **Total** | **/** | **1,948,544** | **504,425** | **326,609,460** | **647.49** |

## Key Features

### 1. Comprehensive Coverage
- Captures a broad range of proteoform diversity
- Integrates data from multiple authoritative sources
- Covers isoforms, genetic variants, and PTMs

### 2. Standardized Representation
- Utilizes PFMOD encoding for unified token representation
- Implements the `Proteoformer` model tokenizer for consistent tokenization across diverse proteoform data
- Compatible with various proteoform notation systems
- Supports integration with existing ontologies (UniMod, UniProt, PSI-MOD)

### 3. Quality Control
- Multi-level quality control procedures
- Manual curation by domain experts
- Consistent data formatting and annotation

## Download

### Stage 1: UniRef50 Database
The UniRef50 database used for Stage 1 masked language model pretraining can be downloaded from:
- **UniRef50**: https://www.uniprot.org/uniref

### Stage 2: π-Proteoformer Corpus
The π-Proteoformer Corpus used for Stage 2 proteoform-aware pretraining can be downloaded from:

*Download link will coming soon.*

## License

The `Proteoform Corpus` is released under the
**Creative Commons Attribution–NonCommercial 4.0 International License (CC BY-NC 4.0)**.

Commercial use of the dataset, including model training for paid services, commercial products, or industrial applications, is strictly prohibited without permission.
