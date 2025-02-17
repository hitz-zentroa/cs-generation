python lince_parse.py --rootdir $1

for split in train dev test
do
    cat $1/lince_$split.tsv| perl detokenizer.perl -q -l en > $1/lince_$split.detokenized.tsv
    # python commandr_document.py --input_file $1/lince_$split.detokenized.tsv
    python create_parallel_corpus.py --cs_path $1/lince_$split.detokenized.tsv --en_path $1/lince_$split.detokenized.monolingual.tsv
    python postprocess.py --input_file $1/lince_$split.detokenized_parallel.tsv
done

