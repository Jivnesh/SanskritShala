const { data } = require("jquery");

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

var content
var e1
var e2
$( document ).ready(function() {
    content = document.getElementById('out-container-box');
    // console.log("Content:", content)
    e1 = document.createElement("center");
    e2 = document.createElement("div");
    e2.setAttribute("style", "display: inline-block;");
});

function myFunction1(id) {
    document.getElementById(id).style.borderWidth = '2px';
    document.getElementById(id).style.padding = "0px";
}
function myFunction2(id) {
    document.getElementById(id).style.borderWidth = '1px';
    document.getElementById(id).style.padding = "1px";
}
function myFunction3(id_name, classname) {
    var tags = document.getElementsByClassName(classname);
    for (var i = 0; i < tags.length; i++) {
        var tem = "";
        tags.item(i).style.borderWidth = '1px';
        tags.item(i).style.padding = '1px';
        tags.item(i).style.fontWeight = '500';
        tags.item(i).style.backgroundColor = 'rgb(' + '180' + ',' + '232' + ',' + '252' + ')';
    }
    document.getElementById(id_name).style.borderWidth = '3px';
    document.getElementById(id_name).style.padding = '0px';
    document.getElementById(id_name).style.backgroundColor = 'rgb(' + '113' + ',' + '207' + ',' + '19' + ')';
    document.getElementById(id_name).style.fontWeight = '700';
    document.getElementById(id_name).onmouseover = false;
    document.getElementById(id_name).onmouseout = false;
}
function myFunction4(id_name) {
    document.getElementById(id_name).style.display = "inline-block";
}
function myFunction5(id_name) {
    document.getElementById(id_name).style.display = "none";
}

function update_segmentation(){
    dic = {};
    var words = document.getElementsByClassName('draggable_operator ui-draggable ui-draggable-handle words');
    for(var i = 0;i<words.length;i++){
        var len = 0;
        // console.log(words[i].getAttribute("datastatus"))
        
        if(words[i].style.display!='none'){
            if(words[i].getAttribute("datastatus")=="off"){
                alert("Please select all words to download final segmentation");
                console.log("Please select all words")
                return;
            }
            len++;
            var pos = parseInt(getPositionXY(words[i]));
            dic[pos] = words[i]
            var ch = words[i].value;
            // console.log(pos,ch);
            
        }
    }
    // console.log("now right");
    var final_dic = sortKeys(dic);
    var key = Object.keys(final_dic);
    var tags = {}
    var words = [];
    for(var i = 0;i<key.length;i++){
        var ch = final_dic[key[i]].value;
        words.push(ch);
        tags[i] = new Set(strip(final_dic[key[i]].title.match(/[^{\}]+(?=})/g)));
        // console.log(key[i],ch, tags[i]);
    }
    document.getElementById("final-segmentation").innerHTML = "Your final segmentation is : "+words.join(" ")
}

function download_segmentation(){
    // var c = document.getElementById('show-morph');
    // c.style.display = "none";
    dic = {};
    var words = document.getElementsByClassName('draggable_operator ui-draggable ui-draggable-handle words');
    for(var i = 0;i<words.length;i++){
        var len = 0;
        // console.log(words[i].getAttribute("datastatus"))
        
        if(words[i].style.display!='none'){
            if(words[i].getAttribute("datastatus")=="off"){
                alert("Please select all words to download final segmentation");
                console.log("Please select all words")
                return;
            }
            len++;
            var pos = parseInt(getPositionXY(words[i]));
            dic[pos] = words[i]
            var ch = words[i].value;
            // console.log(pos,ch);
            
        }
    }
    // console.log("now right");
    var final_dic = sortKeys(dic);
    var key = Object.keys(final_dic);
    var tags = {}
    var words = [];
    for(var i = 0;i<key.length;i++){
        var ch = final_dic[key[i]].value;
        words.push(ch);
        tags[i] = new Set(strip(final_dic[key[i]].title.match(/[^{\}]+(?=})/g)));
        // console.log(key[i],ch, tags[i]);
    }
    var element = document.createElement('a');
    var text = words.join(" ");
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
    element.setAttribute('download', "segmentation.txt");

    element.style.display = 'none';
    document.body.appendChild(element);

    element.click();

    document.body.removeChild(element);
}

function myFunction6(class_name) {
    tags = document.getElementsByClassName(class_name);
    words = document.getElementsByClassName('words-br');
    var j = 0;
    var i;
    var rows = [];

    const color = 'rgb(' + '113' + ',' + ' 207' + ',' + ' 19' + ')';
    // alert(color.length);
    var t = 0;
    for (i = 0; i < tags.length; i++) {
        const temp = [];
        if (tags.item(i).style.backgroundColor == color && tags.item(i).style.display!="none") {
            // alert("Hello");
            // alert(words.item(j).textContent);

            temp.push(words.item(j).textContent);
            temp.push(tags.item(i).textContent);
            j = j + 1;
            rows.push(temp);
        }
        else if (tags.item(i).style.backgroundColor == color && tags.item(i).style.display=="none") {
            t = j;
            break;
        }
        if(j == words.length)break;
    }
    if (j != words.length) {
        alert("Please select tags for all words. You have not selected tags for"+words.item(t).textContent.slice(8));
    }
    else {
        let csvContent = "data:text/csv;charset=utf-8,";
        // var savefile = "";
        rows.forEach(function (rowArray) {
            let row = rowArray.join(",");
            csvContent += row + "\n";
            // savefile += row+"\n";
        });
        // saveDynamicDataToFile(savefile);
        var encodedUri = encodeURI(csvContent);
        window.open(encodedUri);
    }
}

// function extract_tags(words){
//     $.ajax({
//         url:'ajax/extract_tags/',
//         // post for security reason
//         type: "POST",

//         // data that you will like to return 
//         data: {'words' : words},

//         // what to do when the call is success 
//         success:function(response){
//             data = JSON.parse(response);
//             console.log(data);
//             console.log("Extracting Tags");
//             return data;
//         }
//     });
// }

function go_to_pos(words){
    var line = words.join(" ");
    $.ajax({
        url:'ajax/go_to_pos/',
        // post for security reason
        type: "POST",

        // data that you will like to return 
        data: {
            'line':line
        },

        // what to do when the call is success 
        success:function(response){
            // data = JSON.parse(response);
            // console.log(data);
            // console.log("Extracting Tags");
            // return data;
        }
    });
}

function saveDynamicDataToFile(class_name){
    let savefile = "";
    tags = document.getElementsByClassName(class_name);
    words = document.getElementsByClassName('words-br');
    var j = 0, i, t = 0;
    var rows = [];
    const color = 'rgb(' + '113' + ',' + ' 207' + ',' + ' 19' + ')';

    for (i = 0; i < tags.length; i++) {
        const temp = [];
        if (tags.item(i).style.backgroundColor == color && tags.item(i).style.display!="none") {

            temp.push(words.item(j).textContent);
            temp.push(tags.item(i).textContent);
            j = j + 1;
            rows.push(temp);
        }
        else if (tags.item(i).style.backgroundColor == color && tags.item(i).style.display=="none") {
            t = j;
            break;
        }
        if(j == words.length)break;
    }
    if (j != words.length) {
        alert("Please select tags for all words. You have not selected tags for"+words.item(t).textContent.slice(8));
    }
    else {
        // var savefile = "";
        rows.forEach(function (rowArray) {
            let row = rowArray.join(",");
            savefile += row+"\n";
        });
        // saveDynamicDataToFile(savefile);
    }
    document.getElementById("mymodal").style.display = "inline-block";

    $.ajax({
        url:'ajax/save_to_server/',
        // post for security reason
        type: "POST",
        // data that you will like to return 
        data: {'savefile' : savefile},
        // what to do when the call is success 
        success:function(response){
            console.log("done");
            document.getElementById("mymodal").style.display = "none";
            window.open("http://172.26.173.77:8080", "_blank");
            console.log("window open");
        }
    });
}

function checkAlert(evt) {
    var txt = evt.target.value;
    // alert(txt);
    var str1 = "word-";
    var first = str1.concat(evt.target.id.slice(9));
    var id_name = first.concat('-', txt)
    // alert(id_name);
    let tag = document.getElementById(id_name);
    // console.log(tag);
    tag.style.display = "inline-block";
}

function myFunction(id_name) {
    var tags = document.getElementsByClassName("prob-graph");
    for (var i = 0; i < tags.length; i++) {
        tags.item(i).style.display = "none";
    }
    document.getElementById(id_name).style.display = "inline-block";
}

function mymod() {
    document.getElementById("mymodal").style.display = "inline-block";
}

function myfunc(){
    // var c = document.getElementById('show-morph');
    // c.style.display = "none";
    dic = {};
    var words = document.getElementsByClassName('draggable_operator ui-draggable ui-draggable-handle words');
    for(var i = 0;i<words.length;i++){
        var len = 0;
        if(words[i].style.display!='none'){
            len++;
            var pos = parseInt(getPositionXY(words[i]));
            dic[pos] = words[i]
            var ch = words[i].value;
            // console.log(pos,ch);
            
        }
    }
    // console.log("now right");
    var final_dic = sortKeys(dic);
    var key = Object.keys(final_dic);
    var tags = {}
    var words = [];
    for(var i = 0;i<key.length;i++){
        var ch = final_dic[key[i]].value;
        words.push(ch);
        tags[i] = new Set(strip(final_dic[key[i]].title.match(/[^{\}]+(?=})/g)));
        // console.log(key[i],ch, tags[i]);
    }
    // alert(words);

    // go_to_pos(words);
    var line = words.join(" ")
    var url = "http://cnerg.iitkgp.ac.in/tramp/predict?sen="+line+"&in=IAST&out=Devanagari"+"/"
    // var url = "http://172.29.92.118:4000/predict?sen="+line+"&in=IAST&out=Devanagari"
    window.open(url, '_blank')

    // var final_data = extract_tags(words);

    // To show on the same segmentation tool we can use that
    // makePOSTagger(tags, words);
}

function getPositionXY(element) {
    var rect = element.getBoundingClientRect();
    return rect.x;
}

function sortKeys(obj_1) {
    var key = Object.keys(obj_1).sort(); 
    // console.log(key);
    var temp = {};
      
    for (var i = 0; i < key.length; i++) {
        temp[key[i]] = obj_1[key[i]];
        delete obj_1[key[i]];
    } 

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
    var count = 0;
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
    e3312.setAttribute("id", i+'-'+AllTags[j]);
    e3312.innerHTML = AllTags[j];
    return e3312;
}

function add_e341(i,j){
    var e341 = document.createElement("div");
    let str = "div-tags word-"+i+"-tag";
    e341.setAttribute("class",str);
    e341.setAttribute("style", "background-color: rgb(180, 232, 252) ;display : none;text-align: center; padding :1px");
    e341.setAttribute("id", "word-"+i+"-"+AllTags[j]);
    let para1 = 'word-'+i+'-'+AllTags[j];
    let para2 = 'word-'+i+'-tag';

    e341.onmouseover = function() {
        myFunction1(para1);
    };
    e341.onmouseout = function() {
        myFunction2(para1);
    };
    e341.onclick = function() {
        myFunction3(para1,para2);
    };
    e341.ondblclick = function() {
        myFunction5(para1);
    };
    e341.innerHTML = AllTags[j];
    return e341;
}

function addnew_t(i,j,tag){
    var t = document.createElement("div");
    let str = "div-tags word-"+i+"-tag";
    t.setAttribute("class",str);
    t.setAttribute("style", "background-color: rgb(180, 232, 252);text-align: center; padding :1px");
    t.setAttribute("id", "word-"+i+"-tag-"+j);
    let para1 = 'word-'+i+'-tag-'+j;
    let para2 = 'word-'+i+'-tag';

    t.onmouseover = function() {
        myFunction1(para1);
    };
    t.onmouseout = function() {
        myFunction2(para1);
    };
    t.onclick = function() {
        myFunction3(para1,para2);
    };
    t.ondblclick = function() {
        myFunction5(para1);
    };
    t.innerHTML = tag;
    // console.log(tag);
    // console.log('t is: ',t);
    return t;
}

function addnew(i,tags,words){
    var e3 = document.createElement("div");
    e3.setAttribute("style", "display: block; margin : 30px;");
    e3.setAttribute("class", "position-data");

    var e31 = document.createElement("div");
    e31.setAttribute("style", "display: inline-block; border: 1px solid black;background-color: rgb(38, 213, 152);;border-radius: 4px; margin : 3px;padding :1px");
    e31.setAttribute("class", "words-br");
    e31.setAttribute("id", "worddiv-"+i);
    let param_e31 = 'worddiv-'+i;

    e31.onmouseover = function() {
        myFunction1(param_e31);
    };
    e31.onmouseout = function() {
        myFunction2(param_e31);
    };

    var e311 = document.createElement("span");
    e311.setAttribute("class", "span-word");
    e311.setAttribute("id", "word-"+i);
    e311.innerHTML = words[i];

    e31.appendChild(e311);

    var e32 = document.createElement("div");
    e32.setAttribute("style", "font-size: 15px; display: inline-block;");

    var linebreak = document.createElement("br");

    e32.appendChild(linebreak);
    var linebreak2 = document.createElement("br");
    e32.appendChild(linebreak2);

    var e33 = document.createElement("div");
    e33.setAttribute("style", "margin-top: 4px;");
    e33.setAttribute("class", "add-tags");

    var e331 = document.createElement("select");
    e331.setAttribute("name", "tags");
    e331.setAttribute("class", "select-of-adding-tags");
    e331.setAttribute("id", "add-tags-"+i);
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

    // console.log("e34 is: ",e34);
    // console.log("tags are: ",tags[i]);

    let k = 0;
    for (var it = tags[i].values(), val= null; val=it.next().value; ) {
        // console.log(val);
        let tnew = addnew_t(i,k,val);
        e34.appendChild(tnew);
        // console.log("t is: ",tnew);
        k++;
    }

    e3.appendChild(e31);
    e3.appendChild(e32);
    e3.appendChild(e33);
    e3.appendChild(e34);
    e2.appendChild(e3);
    e1.appendChild(e2);
    // console.log(e1)
    // console.log("Content: ", content)
    content.appendChild(e1);
}

function makePOSTagger(tags,words){
    // console.log("In makePOSTagger");
    // console.log(words.length);
    for(var i = 0;i<words.length;i++){
        addnew(i,tags,words);
    }
    var download = document.createElement("center");
    var download_button = document.createElement("button");
    download_button.setAttribute("type","download");
    download_button.setAttribute("class","submit-download-button");
    download_button.setAttribute("style","margin-left:2px");
    download_button.setAttribute("style","margin-right:2px");
    download_button.onclick = function() {
        myFunction6('div-tags');
    };
    download_button.innerHTML = "Download";


    var getdep_button = document.createElement("button");
    getdep_button.setAttribute("type","download");
    getdep_button.setAttribute("class","submit-download-button");
    getdep_button.setAttribute("style","margin-left:2px");
    getdep_button.setAttribute("style","margin-right:2px");
    getdep_button.onclick = function() {
        saveDynamicDataToFile('div-tags');
    };
    getdep_button.innerHTML = "Get Dependency";
    download.appendChild(download_button);
    download.appendChild(getdep_button);
    content.appendChild(download);
    var lbr = document.createElement("br");
    content.appendChild(lbr);
}