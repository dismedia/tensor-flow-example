import * as d3 from "d3"

// set the dimensions and margins of the graph

export const addHeatmap = (params: { dim: number }) => {

    var margin = {top: 0, right: 0, bottom: 0, left: 0},
        width = 400 - margin.left - margin.right,
        height = 400 - margin.top - margin.bottom;

// append the svg object to the body of the page
    var svg = d3.select("body")
        .append("svg")
        .attr("width", 400)
        .attr("height", 500)
        .append("g")


// Labels of row and columns
    var Xdomain = Array.apply(null, Array(params.dim)).map((x, i) => i.toString());
    var Ydomain = Array.apply(null, Array(params.dim)).map((x, i) => i.toString());

// Build X scales and axis:
    var x = d3.scaleLinear()
        .range([0, width])
        .domain(Xdomain)





// Build X scales and axis:
    var y = d3.scaleLinear()
        .range([height, 0])
        .domain(Ydomain)


// Build color scale
    var myColor = d3.scaleLinear()
        .range(["white", "#000000"] as any)
        .domain([0, 1])

    return {
        update: (data:{x:number,y:number,value:number}[]) => {

            svg.selectAll(".heat")
                .data(data, function (d:any,i) {
                    return i.toString()
                })


                .style("fill", function (d) {
                    //console.log(d.value)
                    return myColor(d.value)
                })


                .enter()
                .append("rect")
                .attr("class", "heat")


                .attr("x", function (d) {

                    return x(d.x)
                })
                .attr("y", function (d) {
                    return y(d.y)
                })
                .attr("width", height/params.dim)
                .attr("height", height/params.dim)

                .style("fill", function (d) {
                    //console.log(d.value)
                    return myColor(d.value)
                })




        }
    }

}
