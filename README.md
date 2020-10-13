# Voltage Imaging Data Processing


## Directory Structure
* lib   : preprocessing (motion correction and U-Net preprocessing)

## How to Run

#### Build the pre-processing library
```bash
bash build.sh
```

#### Running the pipeline

Make sure to update the `file_params.txt` accordingly.

To execute in file mode:
```bash
python pipeline --file <tag>

#Example
python pipeline --file 018
```

To execute in batch mode:

```bash
python pipeline --batch
```
The results will be stored as per the paths set in `file_params.txt`.