import React, { useEffect, useState } from 'react'
import '../css/Dp.css'
import Xarrow from 'react-xarrows'
import { FaArrowRight, FaArrowLeft, FaUpload } from 'react-icons/fa'
// const tagsValues = ['prayojanam', 'sampraxanam', 'purvakalah', 'samucciwam', 'anuyogi', 'karwqsamanaxikaranam', 'bavalakranasapwami_samanakalah', 'hewuh', 'apaxanam', 'root', 'samboxyah', 'upapaxasambanxah', 'karma', 'karmasamanaxikaranam', 'axikaranam', 'sambanxah', 'waxarwyam', 'kriyaviseranam', 'prawiyogi', 'karanam', 'karwa', 'viseranam', 'rartisambanxah']
const tagsValues = ['anuyogi', 'apaxanam', 'axikaranam', 'bavalakranasapwami_samanakalah', 'hewuh', 'karanam', 'karma', 'karmasamanaxikaranam', 'karwa', 'karwqsamanaxikaranam', 'kriyaviseranam', 'prawiyogi', 'prayojanam', 'purvakalah', 'rartisambanxah', 'root', 'sambanxah', 'samboxyah', 'sampraxanam', 'samucciwam', 'upapaxasambanxah', 'viseranam', 'waxarwyam']
function Dp() {
    let jval = require('../inputfile/data.json')
    const [input, setinput] = useState("This is the new test sentence to test the data");
    const [tag, settag] = useState('prayojanam')
    const [selected, setSelected] = useState({ first: -1, second: -1 })
    const [filedata, setfiledata] = useState(jval)
    const [arrows, setArrows] = useState([])
    const [page, setPage] = useState(1)

    let inputColor = []
    let color = { 'samucciwam': 'rgb(151,151,151)', 'axikaranam': 'rgb(40.892404046057756,195.90212002826695,248.29351969076725)', 'karmasamanaxikaranam': 'rgb(80.76486511906698,52.51440400293031,53.08090307846933)', 'karma': 'rgb(88.5884063088804,203.67910423153953,209.90796543767718)', 'sampraxanam': 'rgb(129.61976034829996,229.08347665891358,173.3997126553231)', 'apaxanam': 'rgb(228.43215159482878,227.79103939439756,164.96817855566414)', 'bavalakranasapwami_samanakalah': 'rgb(62.66904930870843,129.16280789049665,87.1932825655106)', 'karanam': 'rgb(40.934260339109095,211.0082341777622,43.83889586912186)', 'prawiyogi': 'rgb(137.1310137798212,16.568779621860443,74.52970531571357)', 'anuyogi': 'rgb(63.80087183168032,157.6029046976205,78.49846088699618)', 'root': 'rgb(110.55401377865476,44.29834990456047,198.69765003909146)', 'karwqsamanaxikaranam': 'rgb(25.7895756160927,32.093979258717745,26.327904868555837)', 'rartisambanxah': 'rgb(144.3227317950415,144.5790378977648,231.3117286758576)', 'purvakalah': 'rgb(95.2544330750636,184.51382417829535,72.09252948336751)', 'upapaxasambanxah': 'rgb(69.44475016890506,133.2935468459755,83.99722071535241)', 'waxarwyam': 'rgb(61.463645799040926,26.082897978324898,162.25828698139338)', 'hewuh': 'rgb(6.673138254210781,226.72974767628264,91.26865517995041)', 'kriyaviseranam': 'rgb(176.11552945879745,240.75046624369853,234.98471431815648)', 'viseranam': 'rgb(250.28726103194137,103.98357591697282,13.88006310154271)', 'sambanxah': 'rgb(129.19527775529997,158.05623007107332,40.577465250224115)', 'samboxyah': 'rgb(70.12094532173067,152.7382702021216,209.3796755632114)', 'prayojanam': 'rgb(15.663448901199954,189.05485993705983,116.69688888451144)', 'karwa': 'rgb(37.632586745822074,74.82709935392526,79.74058327798153)' }
    if (selected.first != -1 && selected.second != -1) {
        setArrows(val => [...val, { ...selected, tag: tag }])
        setSelected(val => ({ ...val, first: -1, second: -1 }))
    }
    useEffect(() => {

        if (filedata) {
            setinput(filedata[page - 1].input)
            setArrows(filedata[page - 1].arrow)
        }
    }, [page, filedata])
    input.split(' ').map(val => {
        inputColor.push({ word: val, colors: ['transparent'] })
    })
    useEffect(() => {
        setSelected(val => ({ ...val, first: -1, second: -1 }))
        if (!filedata)
            setArrows([])
    }, [input])
    for (let j = 0; j < arrows.length; j++) {

        inputColor[arrows[j].first - 1]?.colors.push(color[arrows[j].tag])
        inputColor[arrows[j].second - 1]?.colors.push(color[arrows[j].tag])
        if (!arrows[j].height) {
            let num = new Set()
            num.add(10); num.add(20); num.add(30); num.add(40); num.add(50); num.add(60); num.add(70); num.add(80); num.add(90); num.add(100); num.add(110); num.add(120); num.add(130)

            for (let l = 0; l < arrows.length; l++) {
                if (arrows[l].height) {
                    let l1 = Math.abs(arrows[j].first - arrows[j].second) + 1,
                        l2 = Math.abs(arrows[l].first - arrows[l].second),
                        jl = Math.abs(Math.max(Math.max(arrows[j].second, arrows[j].first), Math.max(arrows[l].first, arrows[l].second)) - Math.min(Math.min(arrows[j].first, arrows[j].second), Math.min(arrows[l].first, arrows[l].second))) + 1;
                    if (l1 + l2 >= jl) {
                        num.delete(arrows[l].height)
                    }
                }
                arrows[j].height = Math.max(...num)
            }

        }
    }
    return (
        <div className='dp'>
            <section className='main-container padding'>
                <div className='row justify-content-center'>
                    <input onChange={(e) => { setinput(e.target.value) }} className='col-md-6 input' placeholder='Input the data' />
                </div>
                <div className='row justify-content-center'>
                    <label for='file-input' className='file-input-button col-sm-2' ><FaUpload /> </label>
                    <input onChange={() => {
                        var fileToLoad = document.getElementById("file-input").files[0];
                        var fileReader = new FileReader();
                        fileReader.onload = function (fileLoadedEvent) {
                            var jj = fileLoadedEvent.target.result;
                            jj = JSON.parse(jj)
                            setfiledata(jj)
                        };
                        fileReader.readAsText(fileToLoad, "UTF-8");
                    }} type={'file'} id='file-input' style={{ display: 'none' }} />

                    {/* Download button implementation */}
                    <button onClick={() => {
                        let jj;
                        if (filedata) {
                            jj = filedata; jj[page - 1] = { input: input, arrow: arrows }; setfiledata(filedata)
                        }
                        else {
                            jj = [{ input: input, arrow: arrows }]
                        }
                        var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(jj));
                        var downloadAnchorNode = document.createElement('a');
                        downloadAnchorNode.setAttribute("href", dataStr);
                        downloadAnchorNode.setAttribute("download", "datafile.json");
                        document.body.appendChild(downloadAnchorNode);
                        downloadAnchorNode.click();
                        downloadAnchorNode.remove();

                    }} className='submit-button col-sm-2'>Download</button>

                    {/* For getting model prediction Submit button added*/}
                    <button onClick={() => {
                        // let jj = input;
                        // var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(input);
                        
                        

                        // var downloadAnchorNode = document.createElement('a');
                        // downloadAnchorNode.setAttribute("href", dataStr);
                        // downloadAnchorNode.setAttribute("download", "input.txt");
                        // document.body.appendChild(downloadAnchorNode);
                        // downloadAnchorNode.click();
                        // downloadAnchorNode.remove();

                        
                    }} className='submit-button col-sm-2'>Submit</button>
                </div>
            </section>
            <section className='main-container padding'>
                <div className='row justify-content-center'>
                    <div className='col-lg-8'>
                        <div className='tag-container'>
                            {
                                tagsValues.map(val => (
                                    <span onClick={() => {
                                        settag(val)
                                    }} className='tag-values'
                                        style={{ backgroundColor: tag == val ? color[tag] : '', }}
                                    >
                                        {val}
                                    </span>
                                ))
                            }
                        </div>
                        <p className='text-align-left'>
                            Tag Selected: <span style={{ color: color[tag], fontWeight: 600 }}>{tag}</span>
                        </p>
                    </div>
                </div>
            </section>
            <section className='main-container col-lg-8 padding display-container' style={{ height: '' }}>
                <div className='input-container'>
                    {
                        inputColor.map((val, index) => (
                            <span
                                onClick={() => {
                                    if (selected.first != -1) {
                                        selected.first != index + 1 ? setSelected(val => ({ ...val, second: index + 1 })) : setSelected(val => ({ ...val, first: -1 }))
                                    }
                                    else {
                                        setSelected(val => ({ ...val, first: index + 1 }))
                                    }
                                }}
                                className='input-values' id={index + 1} style={{ color: 'black', backgroundColor: val.colors.length == 1 ? val.colors : '', background: val.colors.length > 1 ? `linear-gradient(45deg,${val.colors})` : '' }}>{val.word}</span>
                        ))
                    }
                </div>
                {
                    arrows.map(val => (
                        <Xarrow start={val.first + ''} end={val.second + ''}
                            startAnchor='top' endAnchor='top' path='curve'
                            color={color[val.tag]}
                            divContainerProps={{
                                style: { position: 'relative' }
                            }}
                            arrowBodyProps={{
                                onClick: () => {
                                    let jj = []
                                    for (let j = 0; j < arrows.length; j++) {
                                        if (arrows[j] != val) {
                                            jj.push(arrows[j]);
                                        }
                                    }
                                    setArrows(val => (jj))
                                },
                                cursor: 'pointer'
                            }}
                            labels={<div style={{
                                color: 'white', backgroundColor: color[val.tag], fontSize: 11, borderRadius: 6, padding: 2, zIndex: 11,
                                position: 'absolute', bottom: (-val.height / 11), display: 'flex', alignItems: 'center'


                            }} >{val.first > val.second ? <FaArrowLeft style={{ display: 'inline' }} /> : ''} {val.tag}  {val.first < val.second ? <FaArrowRight style={{ display: 'inline' }} /> : ''} </div>}
                            _cpy1Offset={-val.height}
                            _cpy2Offset={-val.height}
                            headShape='circle'
                            headSize={4}
                            headColor={'black'}
                        />
                    ))
                }
            </section>
            {filedata && <section className='main-container'>
                <span className='navigation' onClick={() => { if (page > 1) setPage(page - 1); let jj = filedata; jj[page - 1] = { input: input, arrow: arrows }; setfiledata(filedata) }} ><FaArrowLeft size={24} /></span><span>{page}</span><span onClick={() => { if (page < filedata.length) setPage(page + 1); let jj = filedata; jj[page - 1] = { input: input, arrow: arrows }; setfiledata(filedata) }} className='navigation'><FaArrowRight size={24} /> </span>
            </section>}
        </div>
    )
}

export default Dp
