const AllTags = ['abs.',
 'act.',
 'adv.',
 'adv.',
 'aor. [1] ac. du. 1',
 'aor. [1] ac. du. 2',
 'aor. [1] ac. du. 3',
 'aor. [1] ac. pl. 1',
 'aor. [1] ac. pl. 2',
 'aor. [1] ac. pl. 3',
 'aor. [1] ac. sg. 1',
 'aor. [1] ac. sg. 2',
 'aor. [1] ac. sg. 3',
 'aor. [1] md. pl. 3',
 'aor. [1] md. sg. 1',
 'aor. [1] md. sg. 2',
 'aor. [1] md. sg. 3',
 'aor. [1] ps. sg. 3',
 'aor. [2] ac. du. 1',
 'aor. [2] ac. du. 2',
 'aor. [2] ac. du. 3',
 'aor. [2] ac. pl. 1',
 'aor. [2] ac. pl. 2',
 'aor. [2] ac. pl. 3',
 'aor. [2] ac. sg. 1',
 'aor. [2] ac. sg. 2',
 'aor. [2] ac. sg. 3',
 'aor. [2] md. sg. 1',
 'aor. [2] md. sg. 3',
 'aor. [3] ac. pl. 3',
 'aor. [3] ac. sg. 1',
 'aor. [3] ac. sg. 2',
 'aor. [3] ac. sg. 3',
 'aor. [4] ac. du. 2',
 'aor. [4] ac. du. 3',
 'aor. [4] ac. pl. 1',
 'aor. [4] ac. pl. 2',
 'aor. [4] ac. pl. 3',
 'aor. [4] ac. sg. 1',
 'aor. [4] ac. sg. 2',
 'aor. [4] ac. sg. 3',
 'aor. [4] md. pl. 1',
 'aor. [4] md. sg. 1',
 'aor. [4] md. sg. 2',
 'aor. [4] md. sg. 3',
 'aor. [5] ac. pl. 2',
 'aor. [5] ac. sg. 1',
 'aor. [5] ac. sg. 2',
 'aor. [5] ac. sg. 3',
 'aor. [5] md. sg. 1',
 'aor. [5] md. sg. 2',
 'aor. [5] md. sg. 3',
 'aor. [6] ac. sg. 3',
 'aor. [7] ac. pl. 2',
 'aor. [7] ac. sg. 1',
 'aor. [7] ac. sg. 2',
 'aor. [7] md. sg. 3',
 'ben. ac. du. 1',
 'ben. ac. pl. 1',
 'ben. ac. sg. 1',
 'ben. ac. sg. 2',
 'ben. ac. sg. 3',
 'ca. abs.',
 'ca. fut. ac. pl. 2',
 'ca. fut. ac. pl. 3',
 'ca. fut. ac. sg. 1',
 'ca. fut. ac. sg. 2',
 'ca. fut. ac. sg. 3',
 'ca. fut. md. sg. 1',
 'ca. imp. ac. du. 1',
 'ca. imp. ac. pl. 1',
 'ca. imp. ac. pl. 2',
 'ca. imp. ac. pl. 3',
 'ca. imp. ac. sg. 2',
 'ca. imp. ac. sg. 3',
 'ca. impft. ac. du. 3',
 'ca. impft. ac. pl. 2',
 'ca. impft. ac. pl. 3',
 'ca. impft. ac. sg. 1',
 'ca. impft. ac. sg. 2',
 'ca. impft. ac. sg. 3',
 'ca. impft. md. du. 3',
 'ca. impft. md. pl. 3',
 'ca. impft. md. sg. 1',
 'ca. impft. md. sg. 3',
 'ca. impft. ps. pl. 3',
 'ca. impft. ps. sg. 1',
 'ca. impft. ps. sg. 2',
 'ca. impft. ps. sg. 3',
 'ca. imp. md. du. 2',
 'ca. imp. md. du. 3',
 'ca. imp. md. pl. 2',
 'ca. imp. md. sg. 1',
 'ca. imp. md. sg. 2',
 'ca. imp. ps. sg. 1',
 'ca. imp. ps. sg. 2',
 'ca. imp. ps. sg. 3',
 'ca. inf.',
 'ca. opt. ac. du. 1',
 'ca. opt. ac. du. 3',
 'ca. opt. ac. pl. 2',
 'ca. opt. ac. pl. 3',
 'ca. opt. ac. sg. 1',
 'ca. opt. ac. sg. 2',
 'ca. opt. ac. sg. 3',
 'ca. opt. md. du. 3',
 'ca. opt. md. sg. 2',
 'ca. opt. md. sg. 3',
 'ca. opt. ps. sg. 3',
 'ca. per. fut. ac. sg. 1',
 'ca. per. fut. ac. sg. 3',
 'ca. pfp. [1]',
 'ca. pfp. [2]',
 'ca. pfp. [3]',
 'ca. pfu. ac.',
 'ca. pp.',
 'ca. ppa.',
 'ca. ppr. ac.',
 'ca. ppr. md.',
 'ca. ppr. ps.',
 'ca. pr. ac. du. 3',
 'ca. pr. ac. pl. 1',
 'ca. pr. ac. pl. 3',
 'ca. pr. ac. sg. 1',
 'ca. pr. ac. sg. 2',
 'ca. pr. ac. sg. 3',
 'ca. pr. md. pl. 3',
 'ca. pr. md. sg. 1',
 'ca. pr. md. sg. 2',
 'ca. pr. md. sg. 3',
 'ca. pr. ps. du. 3',
 'ca. pr. ps. pl. 1',
 'ca. pr. ps. pl. 3',
 'ca. pr. ps. sg. 1',
 'ca. pr. ps. sg. 2',
 'ca. pr. ps. sg. 3',
 'cond. ac. pl. 3',
 'cond. ac. sg. 1',
 'cond. ac. sg. 2',
 'cond. ac. sg. 3',
 'cond. md. sg. 1',
 'conj.',
 'conj.',
 'des. imp. ac. du. 1',
 'des. imp. ac. sg. 1',
 'des. imp. ac. sg. 2',
 'des. impft. md. sg. 1',
 'des. imp. ps. sg. 3',
 'des. opt. ac. sg. 1',
 'des. per. fut. ac. sg. 3',
 'des. pfp. [1]',
 'des. pft. ac. pl. 2',
 'des. pft. ac. pl. 3',
 'des. pft. ac. sg. 1',
 'des. pft. ac. sg. 3',
 'des. pfu. ac.',
 'des. pp.',
 'des. ppr. ac.',
 'des. pr. md. sg. 1',
 'des. pr. md. sg. 3',
 '* du. abl.',
 '* du. acc.',
 '* du. dat.',
 '* du. g.',
 '* du. i.',
 '* du. loc.',
 '* du. nom.',
 'f. du. abl.',
 'f. du. acc.',
 'f. du. dat.',
 'f. du. g.',
 'f. du. i.',
 'f. du. loc.',
 'f. du. nom.',
 'f. du. voc.',
 'f. pl. abl.',
 'f. pl. acc.',
 'f. pl. dat.',
 'f. pl. g.',
 'f. pl. i.',
 'f. pl. loc.',
 'f. pl. nom.',
 'f. pl. voc.',
 'f. sg. abl.',
 'f. sg. acc.',
 'f. sg. dat.',
 'f. sg. g.',
 'f. sg. i.',
 'f. sg. loc.',
 'f. sg. nom.',
 'f. sg. voc.',
 'fut. ac. du. 1',
 'fut. ac. du. 2',
 'fut. ac. du. 3',
 'fut. ac. pl. 1',
 'fut. ac. pl. 2',
 'fut. ac. pl. 3',
 'fut. ac. sg. 1',
 'fut. ac. sg. 2',
 'fut. ac. sg. 3',
 'fut. md. du. 1',
 'fut. md. du. 3',
 'fut. md. pl. 1',
 'fut. md. pl. 3',
 'fut. md. sg. 1',
 'fut. md. sg. 2',
 'fut. md. sg. 3',
 'iic.',
 'iiv.',
 'imp. [10] ac. du. 3',
 'imp. [10] ac. pl. 1',
 'imp. [10] ac. pl. 2',
 'imp. [10] ac. pl. 3',
 'imp. [10] ac. sg. 2',
 'imp. [10] md. pl. 2',
 'imp. [10] md. sg. 2',
 'imp. [10] md. sg. 3',
 'imp. [1] ac. du. 1',
 'imp. [1] ac. du. 2',
 'imp. [1] ac. du. 3',
 'imp. [1] ac. pl. 1',
 'imp. [1] ac. pl. 2',
 'imp. [1] ac. pl. 3',
 'imp. [1] ac. sg. 1',
 'imp. [1] ac. sg. 2',
 'imp. [1] ac. sg. 3',
 'imp. [1] md. du. 3',
 'imp. [1] md. pl. 2',
 'imp. [1] md. pl. 3',
 'imp. [1] md. sg. 1',
 'imp. [1] md. sg. 2',
 'imp. [1] md. sg. 3',
 'imp. [2] ac. du. 1',
 'imp. [2] ac. du. 2',
 'imp. [2] ac. du. 3',
 'imp. [2] ac. pl. 1',
 'imp. [2] ac. pl. 2',
 'imp. [2] ac. pl. 3',
 'imp. [2] ac. sg. 1',
 'imp. [2] ac. sg. 2',
 'imp. [2] ac. sg. 3',
 'imp. [2] md. du. 3',
 'imp. [2] md. pl. 3',
 'imp. [2] md. sg. 1',
 'imp. [2] md. sg. 2',
 'imp. [2] md. sg. 3',
 'imp. [3] ac. du. 1',
 'imp. [3] ac. du. 2',
 'imp. [3] ac. du. 3',
 'imp. [3] ac. pl. 1',
 'imp. [3] ac. pl. 2',
 'imp. [3] ac. pl. 3',
 'imp. [3] ac. sg. 1',
 'imp. [3] ac. sg. 2',
 'imp. [3] ac. sg. 3',
 'imp. [3] md. pl. 3',
 'imp. [3] md. sg. 1',
 'imp. [3] md. sg. 2',
 'imp. [3] md. sg. 3',
 'imp. [4] ac. du. 1',
 'imp. [4] ac. du. 2',
 'imp. [4] ac. du. 3',
 'imp. [4] ac. pl. 1',
 'imp. [4] ac. pl. 2',
 'imp. [4] ac. pl. 3',
 'imp. [4] ac. sg. 1',
 'imp. [4] ac. sg. 2',
 'imp. [4] ac. sg. 3',
 'imp. [4] md. du. 3',
 'imp. [4] md. pl. 2',
 'imp. [4] md. pl. 3',
 'imp. [4] md. sg. 1',
 'imp. [4] md. sg. 2',
 'imp. [4] md. sg. 3',
 'imp. [5] ac. du. 2',
 'imp. [5] ac. pl. 1',
 'imp. [5] ac. pl. 2',
 'imp. [5] ac. pl. 3',
 'imp. [5] ac. sg. 2',
 'imp. [5] ac. sg. 3',
 'imp. [5] md. pl. 2',
 'imp. [5] md. pl. 3',
 'imp. [6] ac. du. 1',
 'imp. [6] ac. du. 2',
 'imp. [6] ac. du. 3',
 'imp. [6] ac. pl. 1',
 'imp. [6] ac. pl. 2',
 'imp. [6] ac. pl. 3',
 'imp. [6] ac. sg. 1',
 'imp. [6] ac. sg. 2',
 'imp. [6] ac. sg. 3',
 'imp. [6] md. du. 3',
 'imp. [6] md. pl. 2',
 'imp. [6] md. sg. 1',
 'imp. [6] md. sg. 2',
 'imp. [6] md. sg. 3',
 'imp. [7] ac. pl. 2',
 'imp. [7] ac. sg. 2',
 'imp. [7] ac. sg. 3',
 'imp. [7] md. pl. 3',
 'imp. [7] md. sg. 1',
 'imp. [7] md. sg. 2',
 'imp. [8] ac. du. 3',
 'imp. [8] ac. pl. 1',
 'imp. [8] ac. pl. 2',
 'imp. [8] ac. pl. 3',
 'imp. [8] ac. sg. 1',
 'imp. [8] ac. sg. 2',
 'imp. [8] ac. sg. 3',
 'imp. [8] md. pl. 2',
 'imp. [8] md. pl. 3',
 'imp. [8] md. sg. 1',
 'imp. [8] md. sg. 2',
 'imp. [8] md. sg. 3',
 'imp. [9] ac. du. 3',
 'imp. [9] ac. pl. 1',
 'imp. [9] ac. pl. 2',
 'imp. [9] ac. pl. 3',
 'imp. [9] ac. sg. 1',
 'imp. [9] ac. sg. 2',
 'imp. [9] ac. sg. 3',
 'imp. [9] md. pl. 2',
 'imp. [9] md. pl. 3',
 'imp. [9] md. sg. 2',
 'imp. [9] md. sg. 3',
 'impft. [10] ac. pl. 2',
 'impft. [10] ac. pl. 3',
 'impft. [10] ac. sg. 1',
 'impft. [10] ac. sg. 2',
 'impft. [10] ac. sg. 3',
 'impft. [10] md. du. 3',
 'impft. [10] md. pl. 3',
 'impft. [10] md. sg. 1',
 'impft. [10] md. sg. 3',
 'impft. [1] ac. du. 1',
 'impft. [1] ac. du. 2',
 'impft. [1] ac. du. 3',
 'impft. [1] ac. pl. 1',
 'impft. [1] ac. pl. 2',
 'impft. [1] ac. pl. 3',
 'impft. [1] ac. sg. 1',
 'impft. [1] ac. sg. 2',
 'impft. [1] ac. sg. 3',
 'impft. [1] md. du. 1',
 'impft. [1] md. du. 3',
 'impft. [1] md. pl. 2',
 'impft. [1] md. pl. 3',
 'impft. [1] md. sg. 1',
 'impft. [1] md. sg. 2',
 'impft. [1] md. sg. 3',
 'impft. [2] ac. du. 1',
 'impft. [2] ac. du. 2',
 'impft. [2] ac. du. 3',
 'impft. [2] ac. pl. 1',
 'impft. [2] ac. pl. 2',
 'impft. [2] ac. pl. 3',
 'impft. [2] ac. sg. 1',
 'impft. [2] ac. sg. 2',
 'impft. [2] ac. sg. 3',
 'impft. [2] md. du. 3',
 'impft. [2] md. pl. 3',
 'impft. [2] md. sg. 1',
 'impft. [2] md. sg. 2',
 'impft. [2] md. sg. 3',
 'impft. [3] ac. du. 2',
 'impft. [3] ac. du. 3',
 'impft. [3] ac. pl. 1',
 'impft. [3] ac. pl. 2',
 'impft. [3] ac. pl. 3',
 'impft. [3] ac. sg. 1',
 'impft. [3] ac. sg. 2',
 'impft. [3] ac. sg. 3',
 'impft. [3] md. sg. 1',
 'impft. [3] md. sg. 3',
 'impft. [4] ac. du. 1',
 'impft. [4] ac. du. 3',
 'impft. [4] ac. pl. 1',
 'impft. [4] ac. pl. 2',
 'impft. [4] ac. pl. 3',
 'impft. [4] ac. sg. 1',
 'impft. [4] ac. sg. 2',
 'impft. [4] ac. sg. 3',
 'impft. [4] md. du. 3',
 'impft. [4] md. pl. 3',
 'impft. [4] md. sg. 1',
 'impft. [4] md. sg. 2',
 'impft. [4] md. sg. 3',
 'impft. [5] ac. du. 1',
 'impft. [5] ac. du. 3',
 'impft. [5] ac. pl. 2',
 'impft. [5] ac. pl. 3',
 'impft. [5] ac. sg. 1',
 'impft. [5] ac. sg. 2',
 'impft. [5] ac. sg. 3',
 'impft. [6] ac. du. 3',
 'impft. [6] ac. pl. 1',
 'impft. [6] ac. pl. 2',
 'impft. [6] ac. pl. 3',
 'impft. [6] ac. sg. 1',
 'impft. [6] ac. sg. 2',
 'impft. [6] ac. sg. 3',
 'impft. [6] md. du. 3',
 'impft. [6] md. pl. 3',
 'impft. [6] md. sg. 1',
 'impft. [6] md. sg. 3',
 'impft. [7] ac. pl. 2',
 'impft. [7] ac. pl. 3',
 'impft. [7] ac. sg. 2',
 'impft. [7] ac. sg. 3',
 'impft. [7] md. pl. 3',
 'impft. [7] md. sg. 1',
 'impft. [7] md. sg. 3',
 'impft. [8] ac. du. 1',
 'impft. [8] ac. du. 3',
 'impft. [8] ac. pl. 1',
 'impft. [8] ac. pl. 2',
 'impft. [8] ac. pl. 3',
 'impft. [8] ac. sg. 1',
 'impft. [8] ac. sg. 2',
 'impft. [8] ac. sg. 3',
 'impft. [8] md. du. 1',
 'impft. [8] md. pl. 3',
 'impft. [8] md. sg. 1',
 'impft. [8] md. sg. 3',
 'impft. [9] ac. du. 3',
 'impft. [9] ac. pl. 2',
 'impft. [9] ac. pl. 3',
 'impft. [9] ac. sg. 1',
 'impft. [9] ac. sg. 2',
 'impft. [9] ac. sg. 3',
 'impft. [9] md. pl. 3',
 'impft. [9] md. sg. 1',
 'impft. [9] md. sg. 3',
 'impft. ps. du. 3',
 'impft. ps. pl. 3',
 'impft. ps. sg. 1',
 'impft. ps. sg. 2',
 'impft. ps. sg. 3',
 'impft. [vn.] ac. pl. 2',
 'impft. [vn.] ac. sg. 1',
 'impft. [vn.] ac. sg. 2',
 'imp. ps. du. 3',
 'imp. ps. pl. 2',
 'imp. ps. pl. 3',
 'imp. ps. sg. 1',
 'imp. ps. sg. 2',
 'imp. ps. sg. 3',
 'imp. [vn.] ac. du. 1',
 'imp. [vn.] ac. du. 3',
 'imp. [vn.] ac. pl. 1',
 'imp. [vn.] ac. pl. 2',
 'imp. [vn.] ac. sg. 1',
 'imp. [vn.] ac. sg. 2',
 'ind.',
 'ind.',
 'inf.',
 'inj. [1] ac. du. 1',
 'inj. [1] ac. du. 2',
 'inj. [1] ac. du. 3',
 'inj. [1] ac. pl. 1',
 'inj. [1] ac. pl. 2',
 'inj. [1] ac. pl. 3',
 'inj. [1] ac. sg. 1',
 'inj. [1] ac. sg. 2',
 'inj. [1] ac. sg. 3',
 'inj. [1] md. pl. 2',
 'inj. [1] md. pl. 3',
 'inj. [1] md. sg. 1',
 'inj. [1] md. sg. 2',
 'inj. [1] md. sg. 3',
 'inj. [1] ps. sg. 3',
 'inj. [2] ac. pl. 3',
 'inj. [2] ac. sg. 1',
 'inj. [2] ac. sg. 2',
 'inj. [2] ac. sg. 3',
 'inj. [4] ac. pl. 2',
 'inj. [4] ac. sg. 2',
 'inj. [4] ac. sg. 3',
 'inj. [4] md. sg. 1',
 'inj. [4] md. sg. 2',
 'inj. [4] md. sg. 3',
 'int. impft. ac. sg. 2',
 'int. ppr. ac.',
 'int. pr. md. sg. 1',
 'm. du. abl.',
 'm. du. acc.',
 'm. du. dat.',
 'm. du. g.',
 'm. du. i.',
 'm. du. loc.',
 'm. du. nom.',
 'm. du. voc.',
 'm. pl. abl.',
 'm. pl. acc.',
 'm. pl. dat.',
 'm. pl. g.',
 'm. pl. i.',
 'm. pl. loc.',
 'm. pl. nom.',
 'm. pl. voc.',
 'm. sg. abl.',
 'm. sg. acc.',
 'm. sg. dat.',
 'm. sg. g.',
 'm. sg. i.',
 'm. sg. loc.',
 'm. sg. nom.',
 'm. sg. voc.',
 'n. du. abl.',
 'n. du. acc.',
 'n. du. dat.',
 'n. du. g.',
 'n. du. i.',
 'n. du. loc.',
 'n. du. nom.',
 'n. du. voc.',
 'n. pl. abl.',
 'n. pl. acc.',
 'n. pl. dat.',
 'n. pl. g.',
 'n. pl. i.',
 'n. pl. loc.',
 'n. pl. nom.',
 'n. pl. voc.',
 'n. sg. abl.',
 'n. sg. acc.',
 'n. sg. dat.',
 'n. sg. g.',
 'n. sg. i.',
 'n. sg. loc.',
 'n. sg. nom.',
 'n. sg. voc.',
 'opt. [10] ac. pl. 2',
 'opt. [10] ac. pl. 3',
 'opt. [10] ac. sg. 1',
 'opt. [10] ac. sg. 3',
 'opt. [10] md. pl. 3',
 'opt. [10] md. sg. 3',
 'opt. [1] ac. du. 1',
 'opt. [1] ac. du. 2',
 'opt. [1] ac. du. 3',
 'opt. [1] ac. pl. 1',
 'opt. [1] ac. pl. 2',
 'opt. [1] ac. pl. 3',
 'opt. [1] ac. sg. 1',
 'opt. [1] ac. sg. 2',
 'opt. [1] ac. sg. 3',
 'opt. [1] md. du. 3',
 'opt. [1] md. pl. 1',
 'opt. [1] md. pl. 3',
 'opt. [1] md. sg. 1',
 'opt. [1] md. sg. 2',
 'opt. [1] md. sg. 3',
 'opt. [2] ac. du. 1',
 'opt. [2] ac. du. 2',
 'opt. [2] ac. du. 3',
 'opt. [2] ac. pl. 1',
 'opt. [2] ac. pl. 2',
 'opt. [2] ac. pl. 3',
 'opt. [2] ac. sg. 1',
 'opt. [2] ac. sg. 2',
 'opt. [2] ac. sg. 3',
 'opt. [2] md. pl. 3',
 'opt. [2] md. sg. 1',
 'opt. [2] md. sg. 2',
 'opt. [2] md. sg. 3',
 'opt. [3] ac. du. 1',
 'opt. [3] ac. pl. 3',
 'opt. [3] ac. sg. 1',
 'opt. [3] ac. sg. 2',
 'opt. [3] ac. sg. 3',
 'opt. [3] md. sg. 3',
 'opt. [4] ac. du. 1',
 'opt. [4] ac. du. 3',
 'opt. [4] ac. pl. 1',
 'opt. [4] ac. pl. 2',
 'opt. [4] ac. pl. 3',
 'opt. [4] ac. sg. 1',
 'opt. [4] ac. sg. 2',
 'opt. [4] ac. sg. 3',
 'opt. [4] md. du. 3',
 'opt. [4] md. pl. 1',
 'opt. [4] md. pl. 3',
 'opt. [4] md. sg. 1',
 'opt. [4] md. sg. 2',
 'opt. [4] md. sg. 3',
 'opt. [5] ac. du. 1',
 'opt. [5] ac. pl. 1',
 'opt. [5] ac. pl. 3',
 'opt. [5] ac. sg. 1',
 'opt. [5] ac. sg. 2',
 'opt. [5] ac. sg. 3',
 'opt. [5] md. pl. 1',
 'opt. [5] md. sg. 3',
 'opt. [6] ac. du. 1',
 'opt. [6] ac. du. 3',
 'opt. [6] ac. pl. 1',
 'opt. [6] ac. pl. 2',
 'opt. [6] ac. pl. 3',
 'opt. [6] ac. sg. 1',
 'opt. [6] ac. sg. 2',
 'opt. [6] ac. sg. 3',
 'opt. [6] md. pl. 3',
 'opt. [6] md. sg. 2',
 'opt. [6] md. sg. 3',
 'opt. [7] ac. pl. 3',
 'opt. [7] ac. sg. 1',
 'opt. [7] ac. sg. 2',
 'opt. [7] ac. sg. 3',
 'opt. [7] md. pl. 3',
 'opt. [7] md. sg. 2',
 'opt. [7] md. sg. 3',
 'opt. [8] ac. du. 2',
 'opt. [8] ac. du. 3',
 'opt. [8] ac. pl. 1',
 'opt. [8] ac. pl. 3',
 'opt. [8] ac. sg. 1',
 'opt. [8] ac. sg. 2',
 'opt. [8] ac. sg. 3',
 'opt. [8] md. sg. 2',
 'opt. [8] md. sg. 3',
 'opt. [9] ac. pl. 2',
 'opt. [9] ac. pl. 3',
 'opt. [9] ac. sg. 1',
 'opt. [9] ac. sg. 3',
 'opt. [9] md. pl. 2',
 'opt. [9] md. sg. 1',
 'opt. [9] md. sg. 3',
 'opt. ps. du. 3',
 'opt. ps. pl. 1',
 'opt. ps. pl. 3',
 'opt. ps. sg. 2',
 'opt. ps. sg. 3',
 'opt. [vn.] ac. du. 1',
 'part.',
 'per. fut. ac. du. 2',
 'per. fut. ac. du. 3',
 'per. fut. ac. pl. 2',
 'per. fut. ac. pl. 3',
 'per. fut. ac. sg. 1',
 'per. fut. ac. sg. 2',
 'per. fut. ac. sg. 3',
 'pfp. [1]',
 'pfp. [2]',
 'pfp. [3]',
 'pft. ac. du. 1',
 'pft. ac. du. 2',
 'pft. ac. du. 3',
 'pft. ac. pl. 1',
 'pft. ac. pl. 2',
 'pft. ac. pl. 3',
 'pft. ac. sg. 1',
 'pft. ac. sg. 2',
 'pft. ac. sg. 3',
 'pft. md. du. 3',
 'pft. md. pl. 3',
 'pft. md. sg. 1',
 'pft. md. sg. 2',
 'pft. md. sg. 3',
 'pfu. ac.',
 'pfu. md.',
 '* pl. abl.',
 '* pl. acc.',
 '* pl. dat.',
 '* pl. g.',
 '* pl. i.',
 '* pl. loc.',
 '* pl. nom.',
 'pp.',
 'ppa.',
 'ppf. ac.',
 'ppf. md.',
 'ppr. [10] ac.',
 'ppr. [10] md.',
 'ppr. [1] ac.',
 'ppr. [1] md.',
 'ppr. [2] ac.',
 'ppr. [2] md.',
 'ppr. [3] ac.',
 'ppr. [3] md.',
 'ppr. [4] ac.',
 'ppr. [4] md.',
 'ppr. [5] ac.',
 'ppr. [5] md.',
 'ppr. [6] ac.',
 'ppr. [6] md.',
 'ppr. [7] ac.',
 'ppr. [7] md.',
 'ppr. [8] ac.',
 'ppr. [8] md.',
 'ppr. [9] ac.',
 'ppr. [9] md.',
 'ppr. ps.',
 'ppr. [vn.] ac.',
 'pr. [10] ac. du. 1',
 'pr. [10] ac. du. 3',
 'pr. [10] ac. pl. 1',
 'pr. [10] ac. pl. 2',
 'pr. [10] ac. pl. 3',
 'pr. [10] ac. sg. 1',
 'pr. [10] ac. sg. 2',
 'pr. [10] ac. sg. 3',
 'pr. [10] md. pl. 1',
 'pr. [10] md. pl. 3',
 'pr. [10] md. sg. 1',
 'pr. [10] md. sg. 2',
 'pr. [10] md. sg. 3',
 'pr. [1] ac. du. 1',
 'pr. [1] ac. du. 2',
 'pr. [1] ac. du. 3',
 'pr. [1] ac. pl. 1',
 'pr. [1] ac. pl. 2',
 'pr. [1] ac. pl. 3',
 'pr. [1] ac. sg. 1',
 'pr. [1] ac. sg. 2',
 'pr. [1] ac. sg. 3',
 'pr. [1] md. du. 1',
 'pr. [1] md. du. 3',
 'pr. [1] md. pl. 1',
 'pr. [1] md. pl. 3',
 'pr. [1] md. sg. 1',
 'pr. [1] md. sg. 2',
 'pr. [1] md. sg. 3',
 'pr. [2] ac. du. 1',
 'pr. [2] ac. du. 2',
 'pr. [2] ac. du. 3',
 'pr. [2] ac. pl. 1',
 'pr. [2] ac. pl. 2',
 'pr. [2] ac. pl. 3',
 'pr. [2] ac. sg. 1',
 'pr. [2] ac. sg. 2',
 'pr. [2] ac. sg. 3',
 'pr. [2] md. du. 3',
 'pr. [2] md. pl. 1',
 'pr. [2] md. pl. 3',
 'pr. [2] md. sg. 1',
 'pr. [2] md. sg. 2',
 'pr. [2] md. sg. 3',
 'pr. [3] ac. du. 3',
 'pr. [3] ac. pl. 1',
 'pr. [3] ac. pl. 3',
 'pr. [3] ac. sg. 1',
 'pr. [3] ac. sg. 2',
 'pr. [3] ac. sg. 3',
 'pr. [3] md. du. 3',
 'pr. [3] md. pl. 3',
 'pr. [3] md. sg. 1',
 'pr. [3] md. sg. 2',
 'pr. [3] md. sg. 3',
 'pr. [4] ac. du. 1',
 'pr. [4] ac. du. 3',
 'pr. [4] ac. pl. 1',
 'pr. [4] ac. pl. 2',
 'pr. [4] ac. pl. 3',
 'pr. [4] ac. sg. 1',
 'pr. [4] ac. sg. 2',
 'pr. [4] ac. sg. 3',
 'pr. [4] md. du. 1',
 'pr. [4] md. du. 3',
 'pr. [4] md. pl. 1',
 'pr. [4] md. pl. 3',
 'pr. [4] md. sg. 1',
 'pr. [4] md. sg. 2',
 'pr. [4] md. sg. 3',
 'pr. [5] ac. du. 3',
 'pr. [5] ac. pl. 1',
 'pr. [5] ac. pl. 3',
 'pr. [5] ac. sg. 1',
 'pr. [5] ac. sg. 2',
 'pr. [5] ac. sg. 3',
 'pr. [5] md. du. 3',
 'pr. [5] md. pl. 3',
 'pr. [5] md. sg. 2',
 'pr. [5] md. sg. 3',
 'pr. [6] ac. du. 1',
 'pr. [6] ac. du. 2',
 'pr. [6] ac. du. 3',
 'pr. [6] ac. pl. 1',
 'pr. [6] ac. pl. 2',
 'pr. [6] ac. pl. 3',
 'pr. [6] ac. sg. 1',
 'pr. [6] ac. sg. 2',
 'pr. [6] ac. sg. 3',
 'pr. [6] md. du. 1',
 'pr. [6] md. du. 3',
 'pr. [6] md. pl. 1',
 'pr. [6] md. pl. 3',
 'pr. [6] md. sg. 1',
 'pr. [6] md. sg. 2',
 'pr. [6] md. sg. 3',
 'pr. [7] ac. du. 3',
 'pr. [7] ac. pl. 3',
 'pr. [7] ac. sg. 2',
 'pr. [7] ac. sg. 3',
 'pr. [7] md. du. 3',
 'pr. [7] md. pl. 1',
 'pr. [7] md. pl. 3',
 'pr. [7] md. sg. 1',
 'pr. [7] md. sg. 2',
 'pr. [7] md. sg. 3',
 'pr. [8] ac. du. 1',
 'pr. [8] ac. du. 3',
 'pr. [8] ac. pl. 1',
 'pr. [8] ac. pl. 2',
 'pr. [8] ac. pl. 3',
 'pr. [8] ac. sg. 1',
 'pr. [8] ac. sg. 2',
 'pr. [8] ac. sg. 3',
 'pr. [8] md. du. 3',
 'pr. [8] md. pl. 1',
 'pr. [8] md. pl. 3',
 'pr. [8] md. sg. 1',
 'pr. [8] md. sg. 2',
 'pr. [8] md. sg. 3',
 'pr. [9] ac. du. 1',
 'pr. [9] ac. du. 3',
 'pr. [9] ac. pl. 1',
 'pr. [9] ac. pl. 2',
 'pr. [9] ac. pl. 3',
 'pr. [9] ac. sg. 1',
 'pr. [9] ac. sg. 2',
 'pr. [9] ac. sg. 3',
 'pr. [9] md. pl. 1',
 'pr. [9] md. pl. 3',
 'pr. [9] md. sg. 1',
 'pr. [9] md. sg. 2',
 'pr. [9] md. sg. 3',
 'prep.',
 'prep.',
 'pr. ps. du. 1',
 'pr. ps. du. 3',
 'pr. ps. pl. 1',
 'pr. ps. pl. 3',
 'pr. ps. sg. 1',
 'pr. ps. sg. 2',
 'pr. ps. sg. 3',
 'pr. [vn.] ac. du. 3',
 'pr. [vn.] ac. sg. 2',
 'pr. [vn.] ac. sg. 3',
 'pr. [vn.] md. sg. 2',
 '* sg. abl.',
 '* sg. acc.',
 '* sg. dat.',
 '* sg. g.',
 '* sg. i.',
 '* sg. loc.',
 '* sg. nom.',
 '* sg. voc.',
 'tasil']

var content = document.getElementById('out-container-box');
var e1 = document.createElement("center");
var e2 = document.createElement("div");
e2.setAttribute("style", "display: inline-block;");

function myfunc(){
    var c = document.getElementById('show-morph');
    c.style.display = "none";
    dic = {};
    var words = document.getElementsByClassName('draggable_operator ui-draggable ui-draggable-handle words');
    for(var i = 0;i<words.length;i++){
        var len = 0;
        if(words[i].style.display!='none'){
            len++;
            var pos = parseInt(getPositionXY(words[i]));
            dic[pos] = words[i]
            var ch = words[i].value;
            console.log(pos,ch);
            
        }
    }
    console.log("now right");
    var final_dic = sortKeys(dic);
    var key = Object.keys(final_dic);
    var tags = {}
    var words = [];
    for(var i = 0;i<key.length;i++){
        var ch = final_dic[key[i]].value;
        words.push(ch);
        tags[i] = new Set(strip(final_dic[key[i]].title.match(/[^{\}]+(?=})/g)));
        console.log(key[i],ch, tags[i]);
    }
    
    // var writ = document.getElementById('demo');
    // writ.innerHTML = 'Hello How are you?';
    makePOSTagger(tags,words);
}

function getPositionXY(element) {
    var rect = element.getBoundingClientRect();
    return rect.x;
}

function sortKeys(obj_1) {
    var key = Object.keys(obj_1).sort(); 
    console.log(key);
    // Taking the object in 'temp' object
    // and deleting the original object.
    var temp = {};
      
    for (var i = 0; i < key.length; i++) {
        temp[key[i]] = obj_1[key[i]];
        delete obj_1[key[i]];
    } 

    // Copying the object from 'temp' to 
    // 'original object'.
    for (var i = 0; i < key.length; i++) {
        obj_1[key[i]] = temp[key[i]];
    } 
    return obj_1;
}
function strip(arr){
    for(var i = 0;i<arr.length;i++){
        arr[i] = arr[i].substring(1,arr[i].length-1);
    }
    return arr;
}

function distinct(array){
    
           
    var outputArray = [];

    // Count variable is used to add the
    // new unique value only once in the
    // outputArray.
    var count = 0;

    // Start variable is used to set true
    // if a repeated duplicate value is 
    // encontered in the output array.
    var start = false;

    for (j = 0; j < array.length; j++) {
    for (k = 0; k < outputArray.length; k++) {
        if ( array[j] == outputArray[k] ) {
            start = true;
        }
    }
    count++;
    if (count == 1 && start == false) {
        outputArray.push(array[j]);
    }
    start = false;
    count = 0;
    }
    return outputArray;
}

function add_e3312(i,j){
    var e3312 = document.createElement("option");
    e3312.setAttribute("value", AllTags[j]);
    e3312.setAttribute("id", i.toString+'-'+AllTags[j]);
    return e3312;
}

function add_e341(i,j){
    var e341 = document.createElement("div");
    e341.setAttribute("class","div-tags word-"+i.toString+"-tag");
    e341.setAttribute("style", "background-color: rgb(180, 232, 252) ;display : none;text-align: center; padding :1px");
    e341.setAttribute("id", "word-"+i.toString+"-"+AllTags[j]);
    e341.setAttribute("onmouseover", "myFunction1('word-'+i.toString+'-'+AllTags[j]);");
    e341.setAttribute("onmouseout", "myFunction2('word-'+i.toString+'-'+AllTags[j]);");
    e341.setAttribute("onclick", "myFunction3('word-'+i.toString+'-'+AllTags[j],'word-'+i.toString+'-tag');");
    e341.setAttribute("ondblclick", "myFunction5('word-'+i.toString+'-'+AllTags[j]);");
    e341.innerHTML = AllTags[j];
    return e341;
}

function addnew_t(i,j,tags){
    var t = document.createElement("div");
    t.setAttribute("style", "background-color: rgb(180, 232, 252);text-align: center; padding :1px");
    t.setAttribute("id", "word-"+i.toString+"-tag-"+j.toString);
    t.setAttribute("onmouseover", "myFunction1('word-'+i.toString+'-tag-'+j.toString);");
    t.setAttribute("onmouseout", "myFunction2('word-'+i.toString+'-tag-'+j.toString);");
    t.setAttribute("onclick", "myFunction3('word-'+i.toString+'-tag-'+j.toString,'word-'+i.toString+'-tag');");
    t.setAttribute("ondblclick", "myFunction5('word-'+i.toString+'-tag-'+j.toString);");
    t.innerHTML = tags[i][j];
    return t;
}

function addnew(i,tags,words){
    var e3 = document.createElement("div");
    e3.setAttribute("style", "display: block; margin : 30px;");
    e3.setAttribute("class", "position-data");

    

    var e31 = document.createElement("div");
    e31.setAttribute("style", "display: inline-block; border: 1px solid black;background-color: rgb(38, 213, 152);;border-radius: 4px; margin : 3px;padding :1px");
    e31.setAttribute("class", "words");
    e31.setAttribute("id", "worddiv-"+i.toString);
    e31.setAttribute("onmouseover", "myFunction1('worddiv-'+i.toString);");
    e31.setAttribute("onmouseout", "myFunction2('worddiv-'+i.toString);");

    

    var e311 = document.createElement("span");
    e311.setAttribute("class", "span-word");
    e311.setAttribute("id", "word-"+i.toString);
    e311.innerHTML = words[i];

    e31.appendChild(e311);

    var e32 = document.createElement("div");
    e32.setAttribute("style", "font-size: 15px; display: inline-block;");
    e32.setAttribute("class", "words");

    

    var linebreak = document.createElement("br");

    e32.appendChild(linebreak);
    e32.appendChild(linebreak);

    var e33 = document.createElement("div");
    e33.setAttribute("style", "margin-top: 4px;");
    e33.setAttribute("class", "add-tags");

    

    var e331 = document.createElement("select");
    e331.setAttribute("name", "tags");
    e331.setAttribute("class", "select-of-adding-tags");
    e331.setAttribute("id", "add-tags-"+i.toString);
    e331.setAttribute("onclick", "checkAlert(event)");

    
    
    var e3311 = document.createElement("option");
    e3311.setAttribute("value", "choose");
    e3311.innerHTML = "Choose";

    e331.appendChild(e3311);

    
    for(var j = 0;j<AllTags.length;j++){
        e331.appendChild(add_e3312(i,j));
    }

    e33.appendChild(e331);

    var e34 = document.createElement("div");
    e34.setAttribute("style", "display: inline-block; margin-left: 100px; float : right; min-width: 650px;");


    for(var j = 0;j<AllTags.length;j++){
        e34.appendChild(add_e341(i,j));
    }

    
    for(var j = 0;j<tags[i].length;j++){
        e34.appendChild(addnew_t(i,j,tags));
    }
    e3.appendChild(e31);
    e3.appendChild(e32);
    e3.appendChild(e33);
    e3.appendChild(e34);
    e2.appendChild(e3);
    e1.appendChild(e2);
    content.appendChild(e1);
}


function makePOSTagger(tags,words){
    for(var i = 0;i<words.length;i++){
        addnew(i,tags,words);
    }
}