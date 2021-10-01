# Claim Detection & Semantics Extraction (Covid-19)

## Installation

1. Clone repo
2. Create virtual environment
3. Run `bash setup.sh`

## Usage

`python -m claim_detection.models --input TXT_FILES --patterns claim_detection/topics.json --out OUT_FILE`

## Including AMR graphs

In order to use AMR graphs to enhance the results, you will need to run
an action-pointer model from IBM's Transition AMR Parser on the input files.

1. Clone the repository and `cd` into it.
2. Create a new conda environment for the parser (there are several
   conflicting packages between this and the claims detection).
3. Activate the new conda env and run `python -m pip install -e .`
4. Run `bash tests/correclty_installed.sh` to confirm that the installation succeeded.
5. For a basic test, run `bash tests/minimal_test.sh`
6. You will also need the install the alignment tools with these steps:
   1. In the `preprocess/` folder, git clone https://github.com/jflanigan/jamr.git
   2. Make sure that your Java environment is Java 8.
   3. Create the file `~/.sbt/repositories` (if it doesn't already exist)
      and add these lines to it:
      ```
      [repositories]
         maven-central: https://repo1.maven.org/maven2/
      ```
   4. In `jamr/build.sbt`, remove these lines:
      * `import AssemblyKeys._`
      * `assemblySettings`
   5. In the same file, change the `scala-arm` version to 2.0 like so:
      ```
      libraryDependencies ++= Seq(
          "com.jsuereth" %% "scala-arm" % "2.0",
      ```
   6. In `jamr/project/plugins.sbt`, do the following:
      * Update the `sbt-assembly` version (--> 0.14.6)
      * Update the `sbteclipse-plugin` version (--> 5.2.4)
      * Remove the line that adds the `sbt-idea` plugin
   7. Create the file `jamr/project/build.properties` and add an available sbt version to it, like so:
      * `sbt.version=1.2.0`
   8. Within the jamr root, run `./setup`
   9. Still in jamr, run `. scripts/config.sh`
   10. Follow the `# Kevin aligner` steps in `install_alignment_tools.sh`
7. Either train your own model or obtain the data from someone on the project.
8. Run the following:
   ```
   python -m amr_parsing.parse_files \
       TRANSITION_AMR_PARSER_PATH TXT_FILES AMR_OUTPUT_DIR
   ```
9. To run the claims detection with the AMR output, run `claim_detection.models`
   along with the argument `--amrs AMR_OUTPUT_DIR`

