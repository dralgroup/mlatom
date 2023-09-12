#!/bin/bash

function svn_add(){
    cd ../MLatom
    svn add $1
    cd -
}

function svn_del(){
    cd ../MLatom
    svn rm $1
    # rm -r $1
    cd -
}

export -f svn_add
export -f svn_del

for ver in `seq 460 479`; do
    echo start submition: r$ver :
    svn up -r $ver
    cp -r * ../MLatom/
    info=$(svn log --verbose 2> /dev/null | sed -n '/^r'"$ver"'/,/----/p' | sed -n 1p | awk -F '|' '{print " |"$2 $3 "| "}')
    svn log --verbose 2> /dev/null | sed -n '/^r'"$ver"'/,/----/p' | sed -n '/  A /p' | awk '{print $2}' | sed -E 's#^/##' | xargs -i bash -c 'svn_add "$@"' _ {}
    svn log --verbose 2> /dev/null | sed -n '/^r'"$ver"'/,/----/p' | sed -n '/  D /p' | awk '{print $2}' | sed -E 's#^/##' | xargs -i bash -c 'svn_del "$@"' _ {}
    submit=$(svn log --verbose 2> /dev/null | sed -n '/^r'"$ver"'/,/----/p' | tail -n 2 | sed -n 1p)
    cd ../MLatom
    svn up
    svn commit -m "$submit $info"
    svn up
    echo -e "end of r$ver $info $submit \n \n" 
    cd -
done
