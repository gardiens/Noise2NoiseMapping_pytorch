

<a name="readme-top"></a>
<!--





<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://gitlab-student.centralesupelec.fr/alix.chazottes/fmr-2024-segmentation-hierarchique">
    <img src="images/logo_safe.jpg" alt="Logo" width=600>
  </a>

<h3 align="center">  Research Project Template </h3>

  <p align="center">
     Fast train with hydra and lightning
    <br />
    <a href="https://gitlab-student.centralesupelec.fr/alix.chazottes/fmr-2024-segmentation-hierarchique"><strong>Explore the docs »</strong></a>

  </p>
</div>


# First point of View
- Dnas le code y'a pas de Hash  function
- Loss un peu nouveau avec Earth mover Distance
- Code impossible à lire

Lien rapport : https://plmlatex.math.cnrs.fr/7377846448xdjptfpmwjkd

## Initial mail
Bonjour,

Pour le choix des projets, vous trouverez plusieurs documents sur l’espace « supports pédagogiques » sur le site Web du cours : https://www.caor.minesparis.psl.eu/presentation/cours-npm3d-supports/

-       Accès à une liste de projets

-       Liste de bases de données de nuages de points 3D

-       Accès à Google Cloud Platform : tutoriel (voir ci-dessous)

1.     Envoyer une liste de 3 articles demandés, par ordre de priorité, par binôme, d’ici le 7/02, par courriel à mva.npm3d@gmail.com, avec pour sujet [Projet].

Habituellement les projets sont en individuel. Vu le nombre d’inscrits au cours cette année, il sera prioritairement en BINOME.

Les attributions se font sur la base « premier arrivé, premier servi ».

Une demande de projet individuel est recevable si elle est justifiée.

Un correspondant sera associé à chaque projet. Le correspondant a pour rôle d'aider les élèves au démarrage et sera l'évaluateur principal, mais n'est pas un tuteur (sollicitations limitées). Un rapport de démarrage de projet est à envoyer avant le 27 février au correspondant de projet.

Il est possible de proposer un projet sur proposition personnelle. Dans ce cas, envoyer la proposition (article, description) pour validation. Les critères de recevabilité d’un projet sont d’être basés sur un article de recherche, lié au contenu du cours, disponible en ligne, ou bien « classique » ou bien « récent ». Les codes peuvent être disponibles, ou non (ce point doit être clairement identifié). Les articles doivent être reproductibles dans un temps limité compatible avec la durée du projet.

 

2. Envoyer le rapport+code avant le 20/03 dans un fichier .zip à l'adresse mva.npm3d@gmail.com et à l'adresse du correspondant de projet

Instructions :
Faire l'étude d'un article de recherche et tester une implémentation d'une partie ou de la totalité des contributions de l'article

 - Rapport en français ou en anglais
 - 10 pages maximum
 - Etre synthétique sur la description de la méthode
 - Importance de la qualité de la présentation
 - Important d'apporter son analyse critique du ou des contributions scientifiques
 - Implémentation qui doit montrer un résultat sur un ou plusieurs datasets

 

Notation

L’évaluation et la notation se feront sur la base du rapport et du code envoyés, par un jury composé d’intervenants du cours, sur la proposition du correspondant de projet. Le jury se réunira en avril.

 

Les critères de notation (sur 20) sont :

Compréhension de l’article / 5

Implémentation et tests / 5

Propositions d’améliorations / 5

Rédaction du rapport / 5

 

Le projet compte pour 0,7% de la note finale du cours.

 

 

Liste de bases de données : voir sur le site Web du cours

 

 

Puissance de calcul (GPU)

 

Pour les projets, certains ont besoin d’une forte puissance de calcul (ceux-ci sont normalement identifiés sur la liste des projets).

 

Comme chaque élève n’a pas de GPU dédié aux calculs, nous mettons à disposition des crédits Google Cloud pour réaliser des calculs sur une machine Google (sous réserve de confirmation du programme Google Cloud).

 

Important ! Pour l’adresse mail donnée, utilisez votre mail d’établissement.

 

 

Cordialement,

 

François GOULETTE
Et l'équipe pédagogique

# Repository structure
The repository is structured as follows. Each point is detailed below.
```
├── README.md        <- The top-level README for developers using this project
├── configs         <- Configuration files for Hydra. The subtree is detailed below
├── src             <- Source code for use in this project
├── data            <- Data folder, ignored by git
├── logs           <- Logs folder, ignored by git (tensorboard?, wandb, CSVs, ...)
├── venv           <- Virtual environment folder, ignored by git
├── requirements.txt  <- The requirements file for reproducing the analysis environment
├── LICENSE        <- License file
├── train.py         <- Main script to run the code
└── personal_files <- Personal files, ignored by git (e.g. notes, debugging test scripts, ...)
```

This architecture is based on the fact that any research project requires a configuration, possibly decomposed into several sub-configurations


# Setup

## Virtual environment

For the sake of reproducibility, and to avoid conflicts with other projects, it is recommended to use a virtual environment.

There are several ways to create a virtual environment. A good one is Virtual Env and conda.

The following commands create a virtual environment named ``./venv/`` and install the requirements.

```bash
python3 -m venv venv
source venv/bin/activate  # for linux
venv\Scripts\activate.bat  # for windows
pip install -r requirements.txt
#pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 #uncomment if you areon DGX

```

# Setup the script
You just have to follow the script
```sh
bash first_install.sh
#python3 -m venv venv
#source venv/bin/activate  # for linux
#venv\Scripts\activate.bat  # for windows
#pip install -r requirements.txt
#pre-commit install
```

# Configuration system
Use Hydra , see this [doc](docs/hydra.md) for more detail

# clearml
I like clearml because I used it previously and u can plug it on top of every usual loggers, in any case you can fall back to tensorboard if needs be. For a first run you have to do and copy paste what they ask you to do  :

```py
clearml-init

```




# Other tips

## DGX 
I love DGX, the password is the usual as the centraleSupelec one 
```bash
clearml-init

```


## Use Jupyter On a Slurm Cluster
If you want to run Jupyter on a computer node ( the one that has usually GPU).
You should do 
```bash
sbatch script/jupyter.batch
```
Then go to this [notebook](notebooks/NB_cluster.ipynb) and follow instruction 

## Macros

Command line macros are extremely useful to avoid typing the same commands over and over again. This is just a small tip that I like to do, but it can save a lot of time.
## User-personal usefull files

I advice to use files gitignored (there is a `personal_*` field in the `.gitignore` file) to store personal files, such as notes, debugging scripts, etc. It is a good practice to keep the repository clean and organized.


## Disclaimer

I am highly inspired from this awesome [repo](https://github.com/tboulet/research-project-template/tree/main)

# Autotyper

It's something I've been working for a long time I found several options:

- Pytype
- MonkeyType: seems fine if your script is not too slow
-

