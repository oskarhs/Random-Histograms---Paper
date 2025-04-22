using AutoHist, RCall
@rimport histogram

function test_implementation()
    x = randn(5000)
    breaks1 = rcopy(R"""
        h = histogram($x, type="irregular", penalty="penB" verbose=FALSE, plot=FALSE, grid="data")
        h$breaks
    """)
    breaks2 = histogram_irregular(x; rule="penb", grid="data")
    println(breaks1)
    println(breaks2)
end

test_implementation()