
import React, { useState } from "react";
import { BrowserRouter as Router, Route } from "react-router-dom";
import Publications from "../pages/publications";
import Home from "../pages/home";
import Resources from "../pages/Resources";
import Team from "../pages/team";
import Dp from "../pages/Dp";

function AppRouter() {
  const [width,setwidth]=useState(window.innerWidth)
  return (
    <Router basename={"/sanskritshala"}>
      {/* <Route index element={<Home width={width} />} /> */}
      <Route exact path={`/`} component={Home} />
      <Route path={`/publications`} component={Publications} />
      <Route path={`/resources`} component={Resources} />
      <Route path={`/team`} component={Team} />
      <Route path={`/dp`} component={Dp} />
    </Router>
  );
}

export default AppRouter;