function stopMLatom(errorMsg)
    if length(errorMsg) != 0
        println("<!> $errorMsg <!>")
    end 
    exit()
end

function raiseWarning(warningMsg)
    if length(warningMsg) != 0
        println("Warning: $warningMsg")
    end
end 