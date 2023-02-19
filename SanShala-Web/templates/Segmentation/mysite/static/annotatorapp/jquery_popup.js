$(document).ready(function() {
setTimeout(popup, 3000);
//popup function for every action performed
function popup() {
$("#logindiv").css("display", "block");
}
$("#login #cancel").click(function() {
$(this).parent().parent().hide();
});
$("#onclick").click(function() {
$("#contactdiv").css("display", "block");
});
$("#contact #cancel").click(function() {
$(this).parent().parent().hide();
});
// Contact form popup send-button click event.
$("#send").click(function() {
var name = $("#name").val();
var email = $("#email").val();
var contact = $("#contactno").val();
var message = $("#message").val();
if (name == "" || email == "" || contactno == "" || message == ""){
alert("Please Fill All Fields");
}else{
if (validateEmail(email)) {
$("#contactdiv").css("display", "none");
}else {
alert('Invalid Email Address');
}
//function to validate email
function validateEmail(email) {
var filter = /^[w-.+]+@[a-zA-Z0-9.-]+.[a-zA-z0-9]{2,4}$/;
if (filter.test(email)) {
return true;
}else {
return false;
}
}
}
});
// Login form popup login-button click event.
$("#loginbtn").click(function() {
var name = $("#username").val();
var password = $("#password").val();
if (username == "" || password == ""){
alert("Username or Password was Wrong");
}else{
$("#logindiv").css("display", "none");
}
});
});