
import Metal

func sortData() {
    //let device = MTLCreateSystemDefaultDevice()!
    let device = MTLCopyAllDevices()[1]

    let commandQueue = device.makeCommandQueue()!
    let library = device.makeDefaultLibrary()!
    let sortFunction = library.makeFunction(name: "pointBitonic")!
    let pipeline = try! device.makeComputePipelineState(function: sortFunction)

    let shift = 19
    let size = 1<<shift

    typealias DataType = Vector2

    var data: [DataType ] = [DataType](repeating: Vector2(x: 0, y: 0), count: size)
    let squareSize = 1<<(shift >> 1)
    for y in 0..<squareSize {
        for x in 0..<squareSize  {
            data[(y * squareSize) + x].x = Int32(x)
            data[(y * squareSize) + x].y = Int32(y)
        }
    }

    let dataBuffer1 = device.makeBuffer(bytes: &data, length: MemoryLayout<DataType>.stride * data.count, options: [.storageModeManaged])!
    dataBuffer1.didModifyRange(0..<MemoryLayout<DataType>.stride * data.count)
    let threadgroupsPerGrid = MTLSize(width: size/pipeline.threadExecutionWidth, height: 1, depth: 1)
    let threadsPerThreadgroup = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)

    let logn = shift
    let commands = commandQueue.makeCommandBuffer()!

    for i in 0..<logn {
        for j in 0...i {

            var p = UInt32(i)
            var q = UInt32(j)

            let encoder = commands.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(dataBuffer1, offset: 0, index: 0)
            encoder.setBytes(&p, length: MemoryLayout<UInt32>.stride, index: 1)
            encoder.setBytes(&q, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()


        }
    }
    if let blitEncoder = commands.makeBlitCommandEncoder() {
        blitEncoder.synchronize(resource: dataBuffer1)
        blitEncoder.endEncoding()
    }

    var start = Date()
    commands.commit()
    commands.waitUntilCompleted()

    let dataPointer = dataBuffer1.contents().assumingMemoryBound(to: DataType.self)
    let dataBufferPointer = UnsafeMutableBufferPointer(start: dataPointer, count: data.count)
    let resultsArray = Array.init(dataBufferPointer)
    var stop = Date()
    var time =  stop.timeIntervalSince(start)
    print("GPU Time:\(time)")
    start = Date()
    data.sort { (a, b) -> Bool in
        if a.x < b.x {
            return true;
        } else if a.x > b.x {
            return false;
        } else {
            return a.y < b.y;
        }
    }
    stop = Date()
    time = stop.timeIntervalSince(start)
    print("CPU Time:\(time)")

    var match = true
    for i in 0..<data.count where match {
        match = resultsArray[i].x == data[i].x && resultsArray[i].y == data[i].y
    }
    print("shift: \(shift) Match:\(match)")





}

func boundingBox() {
    //let device = MTLCreateSystemDefaultDevice()!
    let device = MTLCopyAllDevices()[1]

    let commandQueue = device.makeCommandQueue()!
    let library = device.makeDefaultLibrary()!
    let sortFunction = library.makeFunction(name: "boundingBox")!
    let pipeline = try! device.makeComputePipelineState(function: sortFunction)

    let shift = 8
    let size = 1<<shift

    typealias DataType = Vector2

    var data: [DataType ] = [DataType](repeating: Vector2(x: 0, y: 0), count: size)
    let squareSize = 1<<(shift >> 1)
    for y in 0..<squareSize {
        for x in 0..<squareSize  {
            data[(y * squareSize) + x].x = Int32(x)
            data[(y * squareSize) + x].y = Int32(y)
        }
    }

    let dataBuffer1 = device.makeBuffer(bytes: &data, length: MemoryLayout<DataType>.stride * data.count, options: [.cpuCacheModeWriteCombined])!

    let threadgroupsPerGrid = MTLSize(width: size/pipeline.threadExecutionWidth, height: 1, depth: 1)
    let threadsPerThreadgroup = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)

    var resultSize: [Int32] = [-1,-1,-1,-1]
    let resultBuf = device.makeBuffer(bytes:&resultSize, length: MemoryLayout<Int32>.stride * resultSize.count, options: [.storageModeManaged])!

    let commands = commandQueue.makeCommandBuffer()!

    let encoder = commands.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(dataBuffer1, offset: 0, index: 0)
    encoder.setBuffer(resultBuf, offset: 0, index: 1)
    encoder.setThreadgroupMemoryLength((pipeline.threadExecutionWidth * 2 * MemoryLayout<Int32>.stride), index: 0)
    encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    if let blitEncoder = commands.makeBlitCommandEncoder() {
        blitEncoder.synchronize(resource: resultBuf)
        blitEncoder.endEncoding()
    }

    var start = Date()
    commands.commit()
    commands.waitUntilCompleted()

    let dataPointer = resultBuf.contents().assumingMemoryBound(to: Int32.self)
    let dataBufferPointer = UnsafeMutableBufferPointer(start: dataPointer, count: 4)
    let resultsArray = Array.init(dataBufferPointer)
    var stop = Date()
    var time =  stop.timeIntervalSince(start)
    print("GPU Time:\(time)")

    print("results:\(resultsArray)")
    let minX = data[Int(resultsArray[0])].x
    let minY = data[Int(resultsArray[1])].y
    let maxX = data[Int(resultsArray[2])].x
    let maxY = data[Int(resultsArray[3])].y

    print("Bounding box:(\(minX),\(minY)), (\(maxX),\(maxY))")
}

boundingBox()
