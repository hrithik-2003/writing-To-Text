const sections = document.querySelectorAll('.section');
const btns = document.querySelectorAll('.controls');
const btn = document.querySelectorAll('.control');
const all = document.querySelectorAll('.main')[0];

function Transition()
{
    // Clicking button - active class

    for (let i = 0 ; i<btn.length; i++ )
    {
        btn[i].addEventListener('click', function(){

            let currentbtn = document.querySelectorAll('.active-btn');
            currentbtn[0].className = currentbtn[0].className.replace('active-btn', '');
            this.className += ' active-btn'

        })
    }

    //Sections - active class

    all.addEventListener('click', (e) => {
        const id = e.target.dataset.id;
        if (id){
            //remove selected from the other buttons
            btns.forEach((button) =>{ 
                button.classList.remove('active')
            })

            e.target.classList.add('active')

            //hide other sections - 

            sections.forEach((section) => {
                section.classList.remove('active');
            })

            const element = document.getElementById(id);
            element.classList.add('active');
        }
    })


    //Lightmode - darkmode

    const themeBtn = document.querySelector('.theme-btn');
    themeBtn.addEventListener('click', () =>{
        let element = document.body;
        element.classList.toggle('light-mode');
    })
}

Transition();