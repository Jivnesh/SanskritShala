import argparse
import os
# /home/jivnesh/Documents/DCST/d
# 'ar','eu','cs','de','hu','lv','pl','sv','fi','ru'
# python utils/io_/convert_ud_to_onto_format.py --ud_data_path d
def write_ud_files(args):
    languages_for_low_resource = ['el']

    languages = sorted(list(set(languages_for_low_resource)))
    splits = ['train', 'dev', 'test']
    lng_to_files = dict((language, {}) for language in languages)
    for language, d in lng_to_files.items():
        for split in splits:
            d[split] = []
        lng_to_files[language] = d
    sub_folders = os.listdir(args.ud_data_path)
    for sub_folder in sub_folders:
        folder = os.path.join(args.ud_data_path, sub_folder)
        files = os.listdir(folder)
        for file in files:
            for language in languages:
                if file.startswith(language) and file.endswith('conllu'):
                    for split in splits:
                        if split in file:
                            full_path = os.path.join(folder, file)
                            lng_to_files[language][split].append(full_path)
                            break

    for language, split_dict in lng_to_files.items():
        for split, files in split_dict.items():
            if split == 'dev' and len(files) == 0:
                files = split_dict['train']
                print('No dev files were found, copying train files instead')
            sentences = []
            num_sentences = 0
            for file in files:
                with open(file, 'r') as file:
                    for line in file:
                        new_line = []
                        line = line.strip()
                        if len(line) == 0:
                            sentences.append(new_line)
                            num_sentences += 1
                            continue
                        tokens = line.split('\t')
                        if not tokens[0].isdigit():
                            continue
                        id = tokens[0]
                        word = tokens[1]
                        pos = tokens[3]
                        ner = tokens[5]
                        head = tokens[6]
                        arc_tag = tokens[7]
                        new_line = [id, word, pos, ner, head, arc_tag]
                        sentences.append(new_line)
            print('Language: %s Split: %s Num. Sentences: %s ' % (language, split, num_sentences))
            if not os.path.exists('data'):
                os.makedirs('data')
            write_data_path = 'data/MRL/ud_pos_ner_dp_' + split + '_' + language
            print('creating %s' % write_data_path)
            with open(write_data_path, 'w') as f:
                for line in sentences:
                    f.write('\t'.join(line) + '\n')

def main():
    # Parse arguments
    args_ = argparse.ArgumentParser()
    args_.add_argument('--ud_data_path', help='Directory path of the UD treebanks.', required=True)

    args = args_.parse_args()
    write_ud_files(args)

if __name__ == "__main__":
    main()
