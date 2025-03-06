// select the tbody element so that it is $0
let c = $0.children;

let index = [
    'Date',
    'Open',
    'High',
    'Low',
    'Close',
    'Adj Close',
    'Volume'
]

for (var i = 0; i < c.length; i++) {
    let child = c[i];
    let subchildren = child.children;

    let output = "";

    for (var ci = 0; ci < subchildren.length; ci++) {
        let subchild = subchildren[ci];
        output += subchild.innerText + ", ";
        // console.log(index[ci] + ": " + subchild.innerText);
    }

    console.log(output)
}

// just copy the contents from the console, paste it into a file, and use regex to clean it