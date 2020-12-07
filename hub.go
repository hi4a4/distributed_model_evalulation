package main

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

const numNodes = 1
const tpIndex = 2
const fpIndex = 3
const fnIndex = 4
const numItems = 5

var wg sync.WaitGroup

// Message lmao
type Message struct {
	nodeID  int
	message int // send a 1 if done
}

// NodeConfigs Struct for storing parameters to the python code
type NodeConfigs struct {
	nodeID       int
	videoName    string
	modelWeights string
	startTime    int
	endTime      int
	masterChan   chan<- Message
}

func getUserArguments() (string, string) {
	if len(os.Args)-1 != 2 {
		log.Fatal("Improper number of arguments, expected 2: <video_name.mp4> <model_weight_file.h5>")
	}
	videoFile := os.Args[1]
	modelWeightsFile := os.Args[2]
	return videoFile, modelWeightsFile
}

func csvReader(filename string) [][]string {
	// 1. Open the file
	recordFile, err := os.Open(filename)
	if err != nil {
		fmt.Println("An error encountered ::", err)
	}
	// 2. Initialize the reader
	reader := csv.NewReader(recordFile)
	// 3. Read all the records
	records, _ := reader.ReadAll()
	// 4. Iterate through the records as you wish
	return records[1:]
}

func combineCsvOutputs(records [][][]string) ([numItems]float64, [numItems]float64, [numItems]float64) {
	var tpAll [numItems]float64
	var fpAll [numItems]float64
	var fnAll [numItems]float64
	var totalPrec [numItems]float64
	var totalReca [numItems]float64
	var totalF1sc [numItems]float64

	for _, rows := range records {
		for i, row := range rows {
			tp, _ := strconv.ParseFloat(row[tpIndex], 64)
			tpAll[i] += tp

			fp, _ := strconv.ParseFloat(row[fpIndex], 64)
			fpAll[i] += fp

			fn, _ := strconv.ParseFloat(row[fnIndex], 64)
			fnAll[i] += fn
		}
	}
	for i := range tpAll {
		totalPrec[i] = tpAll[i] / (tpAll[i] + fpAll[i])
		totalReca[i] = tpAll[i] / (tpAll[i] + fnAll[i])
		totalF1sc[i] = 2 * (totalPrec[i] * totalReca[i]) / (totalReca[i] + totalPrec[i])
	}
	return totalPrec, totalReca, totalF1sc
}

func makeConfigs(videoLength int, videoName string, modelWeights string, numNodes int, masterChan chan Message) []NodeConfigs {
	var configs []NodeConfigs
	var nodeWindow = videoLength / numNodes
	var prevTime = 0
	var endAdd = 0
	var node NodeConfigs
	if videoLength%numNodes != 0 {
		endAdd = videoLength % numNodes
	}
	for i := 0; i < numNodes; i++ {
		if i != numNodes-1 {
			node = NodeConfigs{
				nodeID:       i,
				videoName:    videoName,
				modelWeights: modelWeights,
				startTime:    prevTime,
				endTime:      prevTime + nodeWindow,
				masterChan:   masterChan,
			}
		} else {
			node = NodeConfigs{
				nodeID:       i,
				videoName:    videoName,
				modelWeights: modelWeights,
				startTime:    prevTime,
				endTime:      prevTime + nodeWindow + endAdd,
				masterChan:   masterChan,
			}
		}
		prevTime = prevTime + nodeWindow
		configs = append(configs, node)
	}
	return configs
}

func runNode(node NodeConfigs) {
	var out bytes.Buffer
	var er bytes.Buffer
	cmd := exec.Command("python3.8", "local_predict.py", node.videoName, node.modelWeights, strconv.Itoa(0), strconv.Itoa(28775/32))
	cmd.Stdout = &out
	cmd.Stderr = &er
	err := cmd.Run()
	if err != nil {
		log.Fatal(err)
	}
	
	node.masterChan <- Message{
		nodeID:  node.nodeID,
		message: 1,
	}
	wg.Done()
}

func runNodes(allConfigs []NodeConfigs) {
	wg.Add(len(allConfigs))
	for _, config := range allConfigs {
		go runNode(config)
	}
}

func main() {
	var csvList []string
	var csvRecords [][][]string
	var concepts [numItems]string
	videoFile, modelWeightsFile := getUserArguments()
	fmt.Println(videoFile, modelWeightsFile)
	cmd := exec.Command("python3.8", "get_video_length.py", videoFile)
	var out bytes.Buffer
	var er bytes.Buffer
	var precision, recall, f1score [numItems]float64
	var masterChan chan Message
	masterChan = make(chan Message, 10)
	cmd.Stdout = &out
	cmd.Stderr = &er
	err := cmd.Run()
	if err != nil {
		fmt.Println("failed video length", cmd.Stderr)
		log.Fatal(err)
	}
	numFrames, _ := strconv.ParseFloat(strings.TrimSuffix(out.String(), "\n"), 64)
	fmt.Println("NumFrames: ", numFrames)

	nodeConfigs := makeConfigs(int(numFrames), videoFile, modelWeightsFile, numNodes, masterChan)
	runNodes(nodeConfigs)
	var numMsgs = 0
	var currTime = time.Now()
	for {
		select {
		case msg := <-masterChan:
			if msg.message == 1 {
				numMsgs += 1
			}
		default:
		}
		if time.Now().After(currTime.Add(10*60*time.Second)) {
			log.Fatal("Processing took more than 10 min for at least one node")
		}
	}

	wg.Wait()
	files, err := ioutil.ReadDir("./")
	if err != nil {
		log.Fatal(err)
	}

	for _, f := range files {
		if strings.Contains(f.Name(), ".csv") {
			csvList = append(csvList, f.Name())
		}
	}
	if len(csvList) == 0 {
		log.Fatal("No CSV files")
	}
	for _, file := range csvList {
		csvRecords = append(csvRecords, csvReader(file))
	}
	for i := range concepts {
		concepts[i] = csvRecords[0][i][1]
	}

	precision, recall, f1score = combineCsvOutputs(csvRecords)
	for i := range concepts {
		fmt.Printf("Concept: %s, Precision: %f, recall: %f, f1score: %f\n", concepts[i], precision[i], recall[i], f1score[i])
	}
}
