// const { Callbacks } = require("jquery");

// function callback(url){
//     url = "http://localhost:3000/dp"
//     window.open(url, '_blank')
// }

function gotodependency() {
    var dep_button = document.getElementById("get-dependency")
    dep_button.style.backgroundColor = "#e68484db";
    var loader = document.getElementById("circular-loader")
    loader.style.display = 'inline-block';
    var class_name = 'div-tags';
    tags = document.getElementsByClassName(class_name);
    words = document.getElementsByClassName('words');
    var j = 0;
    var i;
    var rows = [];

    const color = 'rgb(' + '113' + ',' + ' 207' + ',' + ' 19' + ')';
    var t = 0;
    var wordings = [];
    var taggings = []
    for (i = 0; i < tags.length; i++) {
        // const temp = [];
        if (tags.item(i).style.backgroundColor == color && tags.item(i).style.display!="none") {

            wordings.push(words.item(j).textContent.slice(8).trim());
            taggings.push(tags.item(i).textContent);
            j = j + 1;
            // rows.push(temp);
        }
        else if (tags.item(i).style.backgroundColor == color && tags.item(i).style.display=="none") {
            t = j;
            break;
        }
        if(j == words.length)break;
    }
    if (j != words.length) {
        alert("Please select tags for all words. You have not selected tags for"+words.item(t).textContent.slice(8));
        return;
    }
    
    var server_data = [
        {"Words": wordings},
        {"Tags": taggings}
    ];
    var success = false;
    $.ajax({
        type: "POST",
        url: "../gotodependency/",
        data: JSON.stringify(server_data),
        contentType: "application/json",
        dataType: 'json',
        // async:false,
        success: function(result) {
            console.log(result)
            console.log("Hello request is succeed")
            url = "http://cnerg.iitkgp.ac.in/sanskritshala/dp"
            // location.href = url
            window.location = url;
            // window.open(url, '_blank'); 
        } 
    });
}

function gotodep(){
    window.open("http://localhost:3000/dp", '_blank')
}

function get_tags() {
    var class_name = 'div-tags';
    tags = document.getElementsByClassName(class_name);
    words = document.getElementsByClassName('words');
    var j = 0;
    var i;
    var rows = [];

    const color = 'rgb(' + '113' + ',' + ' 207' + ',' + ' 19' + ')';
    var t = 0;
    for (i = 0; i < tags.length; i++) {
        const temp = [];
        if (tags.item(i).style.backgroundColor == color && tags.item(i).style.display!="none") {

            temp.push(words.item(j).textContent.slice(8).trim());
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
        return rows;
    }

}