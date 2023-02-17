#!/bin/bash

for split in dev train test
do
    wget "https://github.com/UniversalDependencies/UD_Serbian-SET/raw/master/sr_set-ud-$split.conllu"
    wget "https://github.com/UniversalDependencies/UD_Croatian-SET/raw/master/hr_set-ud-$split.conllu"
done