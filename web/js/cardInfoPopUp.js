const popUpWin = document.querySelector('.pop-up-wrapper');
const popUpButt = document.querySelector('.button-next');
const popUpExitButt = popUpWin.querySelector('.button-exit-cross');
const popUpPhotos = popUpWin.querySelectorAll('.photo-card');

popUpPhotos.forEach(element => {
    element.addEventListener('mouseenter',(e)=>{
        const del = e.target.querySelector('.delete-hover');
        del.classList.remove('disctivate');
    })

    element.addEventListener('mouseleave',(e)=>{
        const del = e.target.querySelector('.delete-hover');
        del.classList.add('disctivate');
    })
});



function togglePopUp() {
    
    if (popUpWin.classList.contains('disctivate')) {
        popUpWin.classList.remove('disctivate');
        document.body.classList.add('shadow');
    }
    else {
        popUpWin.classList.add('disctivate');
        document.body.classList.remove('shadow');
    }
}

function onHover() {

}