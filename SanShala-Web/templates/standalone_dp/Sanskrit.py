import uuid
from spacy.displacy.templates import (
    TPL_DEP_SVG,
    TPL_DEP_WORDS,
    TPL_DEP_WORDS_LEMMA,
    TPL_DEP_ARCS,
    TPL_ENTS,
)
from spacy.tokens import Doc, Span
from spacy.errors import Errors, Warnings
from spacy.displacy.templates import TPL_ENT, TPL_ENT_RTL, TPL_FIGURE, TPL_TITLE, TPL_PAGE
from spacy.util import minify_html, escape_html, registry
from spacy.errors import Errors
import warnings

TPL_DEP_ARCS = """
<foreignObject class="node" x="{x_div_label}" y="{y_div_label}" width="85" height="{height_labels}">
    <div style="border:1px black solid; text-align: center; font-size: 0.8em; background-color: red; border-radius: 3px; " id = "labeldiv-{id}-{i}">
        <select name="cars" id="selectlabeldiv-{id}-{i}"  style="text-align: center; padding: 2px; font-size: 0.8em; appearance: none; border: 0px; outline: 0px; text-align: center;background-color: red; border-radius: 3px; height : 20px; width: "{width_labels}"; ">
            <option value="volvo">{label}</option>
            <option value="saab">viseranam</option>
            <option value="mercedes">karanam</option>
            <option value="audi">axikaranam</option>
        </select>
    </div>
</foreignObject>
<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-{id}-{i}" stroke-width="{stroke}px" d="{arc}" fill="none" stroke="{currentColor}"/>
    <!--<text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px; font-weight: 600px">
        <textPath xlink:href="#arrow-{id}-{i}" class="displacy-label" startOffset="50%" side="{label_side}" fill="{currentColor}" text-anchor="middle">{label}</textPath>
    </text>-->
    <path class="displacy-arrowhead" d="{head}" fill="{currentColor}"/>
</g>
"""

TPL_DEP_WORDS = """
<foreignObject class="node" x="{x_div}" y="{y_div}" width="{width_words}" height="{height_words}">
    <div style="border:1px black solid; text-align: center; background-color: burlywood; border-radius: 3px;" id = "worddiv-{id}-{i}" onmouseover="bigImg(this)" onmouseout="normalImg(this)" onclick="myFunction(flag)">{text}</div>
</foreignObject>
<text class="displacy-token" fill="currentColor" text-anchor="middle" y="{y}">
    <!--<tspan style=" border: 1px solid black; border-radius: 0.25rem; display : inline-block; padding: 4px;" id = "word-{id}-{i}" class="displacy-word" fill="currentColor" x="{x}" >{text}</tspan>-->
    <tspan style=" border: 1px solid {currentColor}; border-radius: 0.25rem; background-color: white; display : inline-block; color: {currentColor}; padding: 4px;" id = "tag-{id}-{i}" class="displacy-tag" dy="2em" fill="currentColor" x="{x}" >{tag}</tspan>
</text>
"""

TPL_DEP_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:lang="{lang}" id="{id}" class="displacy" width="{width}" height="{height}" direction="{dir}" style="max-width: none; height: {height}px; color: {color}; background: {bg}; font-family: {font}; direction: {dir}">{content}</svg>
"""


DEFAULT_LANG = "en"
DEFAULT_DIR = "ltr"
_html = {}
RENDER_WRAPPER = None



def get_sans_html(filename=None):
    offset_x = 50                                                                          # x-coordinate of starting point of first word in svg box
    distance = 175                                                                         # distance_between_words
    raw_data = open(filename)
    # print("raw data is : ")
    # print(raw_data)
    # print()
    data_string_list = list(raw_data)  
    # print("list is : ")
    # print(data_string_list)      
    # print()      
    while(data_string_list[-1]=='\n'):
        data_string_list.pop()                                
    arrow_stroke = 2                                                                       # stroke-width of arc
    arrow_spacing = 20
    word_spacing = 45
    compact = False
    arrow_width =  8 #if compact else 10
    sentence_length = len(data_string_list)                                                # number of words in the string
    # columns_list = [0, 1, 3, 4, 6, 7]
    data = [(data_string_list[i].strip()).split('\t') for i in range(sentence_length)]
    print(data)
    words = [data[i][1] for i in range(sentence_length)]
    length_words = [len(word) for word in words]
    tags = [data[i][3] for i in range(sentence_length)]
    dep = [data[i][7] for i in range(sentence_length)]
    x = [50+i*175 for i in range(sentence_length)]                                         # x-coordinate of starting point of all words in svg box
    width = offset_x + sentence_length * distance                                          # width of svg box
    id  = uuid.uuid4().hex                                                                 # id for <svg> tag
    color = "#000000"                                                                      # color in <svg> tag
    bg = "#ffffff"                                                                         # background color for <svg> tag
    dir = "ltr"                                                                            # direction inside <svg> tag
    lang = "en"                                                                            # language inside svg tag
    font = "Arial"                                                                         # font inside <svg> tag
    heads = [((int(data[i][6])-1) if int(data[i][6])>0 else 0) for i in range(sentence_length)]                              # heads of words
    dimensions_words = [GetTextDimensions(words[i], 22, font) for i in range(sentence_length)]
    dimensions_labels = [GetTextDimensions(dep[i], 17, font) for i in range(sentence_length)]
    
    # print(heads)
    # if max(length_words)>10:
    #     distance = 250
    colors = ["#FFBF00","#00FFFF","#3DDC84","#3DDC84","#E30022","#6F00FF","#E48400","#CE2029","#8806CE","#228B22","#1034A6","#FF1493","DodgerBlue","Orange"]
    arcs = []
    for i in range(sentence_length):
        if i < heads[i]:
            arcs.append(
                {
                    "start": i,
                    "end": heads[i],
                    "label": dep[i],
                    "dir": "left"
                }
            )
        elif i > heads[i]:
            arcs.append(
                {
                    "start": heads[i],
                    "end": i,
                    "label": dep[i],
                    "dir": "right",
                }
            )
    parsed = {"arcs" : arcs}
    # print(parsed)
    levels = [arc["end"]-arc["start"] for arc in parsed["arcs"]]
    highest_level = max(levels)
    # print(highest_level)
    offset_y = distance / 2 * highest_level + arrow_stroke
    height = offset_y + 3 * word_spacing
    y = offset_y + word_spacing


    # Creating code for words
    rendered_words = []
    for i in range(sentence_length):
        # print(dimensions_words[i])
        s = TPL_DEP_WORDS.format(text=words[i], tag=tags[i],
                                x=x[i], y=y, currentColor=colors[i], id = id, i = i,
                                x_div = x[i]-(dimensions_words[i][0]+8)/2, y_div = y-24,
                                width_words = dimensions_words[i][0]+10, height_words = dimensions_words[i][1]+15 )
        rendered_words.append(s)




    # Creating code for arc
    rendered_arcs = []
    for i in range(sentence_length):
        start = parsed["arcs"][i]["start"]
        end = parsed["arcs"][i]["end"]
        label = parsed["arcs"][i]["label"]
        dir = parsed["arcs"][i]["dir"]
        # if end-start == 1:
        #     compact = True
        # else:
        #     compact = False

        if start < 0 or end < 0:
            error_args = dict(start=start, end=end, label=label, dir=dir)
            raise ValueError(Errors.E157.format(**error_args))
        level = levels[i]
        x_start = offset_x + start * distance + arrow_spacing
        # if dir == "rtl":
        #     x_start = width - x_start
        y = offset_y
        x_end = (
            offset_x
            + (end - start) * distance
            + start * distance
            - arrow_spacing * (highest_level - level) / 4
        )
        dif = x_end-x_start
        y_curve = offset_y - level * distance / 4
        if dir == 'right':
            x_start += arrow_spacing * (highest_level - level) / 4
        arrowhead = get_arrowhead(dir, x_start, y, x_end, arrow_width)
        if label == 'root':
            continue
        
        arc = get_arc(x_start, y, y_curve, x_end, compact)
        label_side = "right" if dir == "left" else "left"
        width_labels = dimensions_labels[i][0] + 15
        if(dif<3):
            width_labels = dimensions_labels[i][0] + 11
        
        rendered_arc =  TPL_DEP_ARCS.format(
            id=id,
            i=i,
            stroke=3,
            head=arrowhead,
            label=label,
            label_side=label_side,
            arc=arc,
            currentColor = colors[i],
            height_labels = dimensions_labels[i][1] + 15,
            width_labels = width_labels,
            x_div_label = (x_start+x_end)/2-(dimensions_labels[i][0])/2-(sentence_length-i)*2.15,
            y_div_label = y_curve + 5
        )
        # rendered_arc.replace('"currentColor"',colors[i])
        # print(rendered_arc)
        # print()
        rendered_arcs.append(rendered_arc)
        # print(rendered_arc)
    # print(rendered_words)
    # print('#######################################################################################################################################################')
    # print(rendered_arcs)
    content = "".join(rendered_words)+"".join(rendered_arcs)
    content = content.replace("\n\n",'\n')
    content = content
    content = content.strip()
    content = content + """<script>
function bigImg(x) {
  x.style.backgroundColor= "rgb(174, 131, 75)";
}

function normalImg(x) {
    x.style.backgroundColor= "burlywood";
}
function myFunction(e){
    if (e==1){
        document.getElementsByClassName("node").innerHTML= "Hello";
        flag = 0;
    }
    else if(e==0){
        document.getElementsByClassName("node").innerHTML = "Hello World"
        flag += 1;
    }
}
</script>"""
    return TPL_DEP_SVG.format(
            id=id,
            width=width,
            height=height,
            color=color,
            bg=bg,
            font=font,
            content=content,
            dir=dir,
            lang=lang,
            )




def get_arc( x_start, y, y_curve, x_end, compact):
    """Render individual arc.

    x_start (int): X-coordinate of arrow start point.
    y (int): Y-coordinate of arrow start and end point.
    y_curve (int): Y-corrdinate of Cubic Bézier y_curve point.
    x_end (int): X-coordinate of arrow end point.
    RETURNS (unicode): Definition of the arc path ('d' attribute).
    """
    template = "M{x},{y} {x},{c} {e},{c} {e},{y}"
    return template.format(x=x_start, y=y, c=y_curve, e=x_end)

def get_arrowhead(direction, x, y, end, arrow_width):
    """Render individual arrow head.

    direction (unicode): Arrow direction, 'left' or 'right'.
    x (int): X-coordinate of arrow start point.
    y (int): Y-coordinate of arrow start and end point.
    end (int): X-coordinate of arrow end point.
    RETURNS (unicode): Definition of the arrow head path ('d' attribute).
    """
    # arrow_width = 10
    if direction == "left":
        pos1, pos2, pos3 = (x, x - arrow_width + 2, x + arrow_width - 2)
    else:
        pos1, pos2, pos3 = (
            end,
            end + arrow_width - 2,
            end - arrow_width + 2,
        )
    arrowhead = (
        pos1,
        y + 2,
        pos2,
        y - arrow_width,
        pos3,
        y - arrow_width,
    )
    return "M{},{} L{},{} {},{}".format(*arrowhead)


# import ctypes

def GetTextDimensions(text, points, font):
    from PIL import ImageFont
    font = font.lower()+'.ttf'
    font = '/usr/share/fonts/truetype/font-awesome/fontawesome-webfont.ttf'
    font = ImageFont.truetype(font, points)
    size = font.getsize(text)
    print("size is :",size)
    x,y = size
    # class SIZE(ctypes.Structure):
    #     _fields_ = [("cx", ctypes.c_long), ("cy", ctypes.c_long)]

    # hdc = ctypes.windll.user32.GetDC(0)
    # hfont = ctypes.windll.gdi32.CreateFontA(points, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, font)
    # hfont_old = ctypes.windll.gdi32.SelectObject(hdc, hfont)

    # size = SIZE(0, 0)
    # ctypes.windll.gdi32.GetTextExtentPoint32A(hdc, text, len(text), ctypes.byref(size))

    # ctypes.windll.gdi32.SelectObject(hdc, hfont_old)
    # ctypes.windll.gdi32.DeleteObject(hfont)


    return [x, y]
# filename = 'test.conll'
# get_sans_html(filename = filename)