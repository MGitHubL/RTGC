# RTGC

This repository provides the PyTorch implementation of RTGC for temporal graph clustering. RTGC models interaction reliability, historical reliability, and clustering reliability within a temporal graph clustering framework.

If you encounter any problems, please open an issue or contact us at `mengliuedu@163.com`.

## Datasets

**Currently, only the BDSCSocial dataset is publicly available for this release. All datasets used in this work will be made publicly available after the paper is accepted.**

For the temporal graph clustering benchmark and related dataset resources, please refer to [BenchTGC](https://github.com/MGitHubL/BenchTGC).

## Prepare

### Dependencies

Install PyTorch for your environment and the following Python packages:

```bash
pip install numpy scipy scikit-learn networkx gensim munkres
```

### Data Preparation

Clone the repository:

```bash
git clone https://github.com/MGitHubL/RTGC.git
cd RTGC
```

Create a `data` directory alongside the `RTGC` directory. For BDSCSocial, prepare the following files. Paths below are relative to the repository root.

| File | Format |
| --- | --- |
| `../data/BDSCSocial/BDSCSocial.txt` | Each row: `source_node target_node timestamp` |
| `../data/BDSCSocial/node2label.txt` | Each row: `node_id label` |

Node IDs must be consecutive integers starting from zero and consistent across the interaction, label, and embedding files.

### Pre-Training

Run the following commands to generate the initial node embeddings:

```bash
cd pretrain
python pretrain.py --data BDSCSocial \
    --input ../../data/BDSCSocial/BDSCSocial.edgelist \
    --output BDSCSocial_feature.emb
cd ..
```

The script converts the interactions into an edge list and generates 128-dimensional node2vec embeddings by default. The resulting file is stored at `pretrain/BDSCSocial_feature.emb`.

You may also provide an existing embedding file in word2vec text format: a header containing the number of nodes and embedding dimension, followed by rows containing `node_id` and its embedding values.

## Run

From the repository root, run:

```bash
python main.py --dataset BDSCSocial --seed 0
```

Dataset configurations and default parameters are defined in `main.py`. Dataset settings are applied after argument parsing, followed by the selected ablation settings. To change the three reliability strengths or learning rate decay switch, edit `DATASET_CONFIGS` or `MODEL_DEFAULTS` in `main.py`.

Use `python main.py --help` to view the available arguments. If multiprocessing causes DataLoader issues in your environment, add `--num_workers 0`.

### Ablation Experiments

Select a variant with `--ablation`:

| Variant | Configuration |
| --- | --- |
| `full` | Full RTGC model; default setting |
| `tgc` | All three reliability strengths set to zero |
| `without_interaction_fuzziness` | Interaction reliability strength set to zero |
| `without_triggering_fuzziness` | Historical reliability strength set to zero |
| `without_membership_fuzziness` | Clustering reliability strength set to zero |

For example:

```bash
python main.py --dataset BDSCSocial --ablation without_triggering_fuzziness --seed 0
```

## Test

Clustering performance is evaluated after each epoch using KMeans. The code reports ACC, NMI, ARI, and macro-F1.

The final summary reports the epoch with the highest arithmetic mean of these four metrics, together with the best epoch for each individual metric.

## License

See [LICENSE](LICENSE).
