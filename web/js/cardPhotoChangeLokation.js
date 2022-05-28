const field = document.querySelector('.pos-name');
const inputField = field.querySelector('.input-field');
const form = field.querySelector('.inputForm');

let temp;

inputField.onblur = onSubmint;


function onSubmint() {
    if(inputField.value.length < 1){
        temp.classList.remove('disactivate');
        temp.classList.add('text-decoration-underline-border-bottom');
        temp.classList.remove('black-text-color');
        temp.textContent = "Задайте название местности";
        inputField.classList.add('disactivate');
    }
    else{
        temp.classList.remove('disactivate');
        temp.classList.remove('text-decoration-underline-border-bottom');
        temp.classList.add('black-text-color');
        temp.textContent = inputField.value;
        inputField.classList.add('disactivate');
    }
}

form.addEventListener('submit', (e)=>{
    e.preventDefault();

    onSubmint();
})

field.addEventListener('click', (e)=>{
    temp = field.querySelector('.pos-name-name');
    temp.classList.add('disactivate');
    inputField.classList.remove('disactivate');
    inputField.focus();
});