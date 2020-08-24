# Essay grading for Orange

This is an essay grading add-on for [Orange data mining](https://orange.biolab.si/).

## Installation

- [Download and install Orange](https://orange.biolab.si/download/)

- Open Orange
    - Click Options > Add-ons
    - Install Orange3-Text add-on

- Clone this repository

- Open Anaconda prompt (as Admin), navigate to cloned repository, activate Orange environment and install essay grading:

        cd essay-grading
        activate [path_to_Orange_environment] (e.g. "C:\Program Files\Orange")
        pip install -e .
		conda install pytorch torchvision cpuonly -c pytorch
		pip install flair==0.4.5

    - If any problems arise during Flair installation, try running the following command before installing Flair

            pip install transformers==2.4.1
        

- In addition to the above, we need to take care of neuralcoref manually. 
[Download neuralcoref from here](https://github.com/huggingface/neuralcoref) navigate to it and run:

        cd neuralcoref
        pip install -r requirements.txt
        pip install -e .

- You will also need to download SpaCy's english language models:

        python -m spacy download en
        python -m spacy download en_core_web_lg 
        python -m spacy download en_vectors_web_lg 
        
        
- DONE!
    - Your code changes will take effect immediately or after restarting Orange (if open).

## Usage

- Run Orange
- All essay grading widgets are bundled in "Essay grading"
- You can find a few examples by clicking Help > Example Workflows



## Remarks

This is an implementation of an essay grading system described in [PhD thesis](http://eprints.fri.uni-lj.si/4133/1/63120364-KAJA_ZUPANC-Semanti%C4%8Dno_usmerjeno_avtomatsko_ocenjevanje_esejev.pdf) by Kaja Zupanc.

AGE and AGE+ systems are fully functional. SAGE is functional but requires a lot of time to complete.