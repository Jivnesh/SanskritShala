import {BrowserRouter as Router, Route, Routes, HashRouter} from 'react-router-dom';
import './App.css';
import Home from '../src/pages/home'
import React, { useState } from 'react';
import WebFont from 'webfontloader';
import Publications from './pages/publications';
import Team from './pages/team';
import JJ from './pages/jj';
import Resources from './pages/Resources'
import Dp from './pages/Dp';
import 'bootstrap/dist/css/bootstrap.min.css'
import Old_Dp from './pages/Old_Dp';
import AppRouter from "./Routes/AppRouter";
// function CC() {
//   // 👇️ redirect to external URL
//   window.location.replace('http://172.29.92.118:4040/');

//   return null;
// }

function App() {
  WebFont.load({
    google:{
      families:['Roboto:300,400,700','Montserrat']
    }
  })
  const [width,setwidth]=useState(window.innerWidth)
  window.addEventListener('resize',()=>{
    setwidth(window.innerWidth)
  })  
//   return (<div className="App"> 
//   <HashRouter> 
//   <Routes>
//       <Route path='/sanskritshala'> {/* 'sanskritshala'*/}
//           <Route index element={<Home width={width} />} />
//           <Route path='/' element={ <Home /> } />
//           <Route path='/publications' element={ <Publications /> } />
//           <Route path='/team' element={ <Team /> } />
//       </Route>
//       {/* <Route path="/*" element={<Navigate to="/sanskritshala" />}  /> navigate to default route if no url matched */}
//   </Routes>
// </HashRouter></div>);
    return (<div className='App' >
      <Router basename="/sanskritshala">
      <Routes>
        {/* <Route index element={<Home width={width} />} /> */}
        {/* <Route path="/sanskritshala"></Route> */}
        <Route path="*" element={<Home />} />
        <Route exact path="/" element={<Home/>}/>
        <Route path="/publications" element={<Publications/> } />
        <Route path="/team" element={<Team/>} />
        
        <Route path="/resources" element={<Resources/>} />
        <Route path='/dp' element={<Dp/>}/>
        {/* <Route path="/CC" element={<CC />} /> */}
      </Routes>
    </Router>
  </div>
  );
  // return (<div className='App' >
  //     <Router basename={"/sanskritshala"}>
  //     <Routes>
  //       {/* <Route index element={<Home width={width} />} /> */}
  //       <Route exact path="/" element={<Home/>}/>
  //       <Route path="/publications" element={<Publications/> } />
  //       <Route path="/team" element={<Team/>} />
        
  //       <Route path="/resources" element={<Resources/>} />
  //       <Route path='/dp' element={<Dp/>}/>
  //       {/* <Route path="/CC" element={<CC />} /> */}
  //     </Routes>
  //   </Router>
  // </div>
  // );
}

export default App;
