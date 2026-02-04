const video = document.getElementById("video");
var pos;
var res;

Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromUri("http://localhost:5001/models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("http://localhost:5001/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("http://localhost:5001/models"),
  faceapi.nets.faceExpressionNet.loadFromUri('http://localhost:5001/models')
]).then(startWebcam);

function startWebcam() {
	navigator.mediaDevices
		.getUserMedia({
		video: true,
		audio: false,
		})
		.then((stream) => {
		video.srcObject = stream;
		})
		.catch((error) => {
		console.error(error);
		});
}

function getLabeledFaceDescriptions() {
	const labels = [userId];
	return Promise.all(
		labels.map(async (label) => {
		const descriptions = [];
		for (let i = 1; i <= 2; i++) {
			const img = await faceapi.fetchImage(`http://localhost:5001/labels/${label}/${i}.jpg`);
			const detections = await faceapi
			.detectSingleFace(img)
			.withFaceLandmarks()
			.withFaceDescriptor()
			.withFaceExpressions();
			descriptions.push(detections.descriptor);
		}
		return new faceapi.LabeledFaceDescriptors(label, descriptions);
		})
	);
}

video.addEventListener("play", async () => {
	const labeledFaceDescriptors = await getLabeledFaceDescriptions();
	//max distance
	const maxDescriptorDistance = 0.35;
	const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, maxDescriptorDistance);

	const canvas = faceapi.createCanvasFromMedia(video);
	document.body.append(canvas);

	const displaySize = { width: video.width, height: video.height };
	faceapi.matchDimensions(canvas, displaySize);

	setInterval(async () => {
		const detections = await faceapi
		.detectAllFaces(video)
		.withFaceLandmarks()
		.withFaceDescriptors()
		.withFaceExpressions();

		const resizedDetections = faceapi.resizeResults(detections, displaySize);

		canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
		//faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
		faceapi.draw.drawFaceExpressions(canvas, resizedDetections)

		const results = resizedDetections.map((d) => {
		return faceMatcher.findBestMatch(d.descriptor);
		});

		results.forEach((result, i) => {
			const box = resizedDetections[i].detection.box;
			const drawBox = new faceapi.draw.DrawBox(box, {
				label: result,
			});
			drawBox.draw(canvas);
			if(result['_label'] !== 'unknown') {
        		closeWebcam();
				res = result['_label'];
				saveAttendance();
			}
		});
	}, 100);
});

function closeWebcam() {
	const video = document.querySelector('video');

	// A video's MediaStream object is available through its srcObject attribute
	const mediaStream = video.srcObject;

	// Through the MediaStream, you can get the MediaStreamTracks with getTracks():
	const tracks = mediaStream.getTracks();

	// Tracks are returned as an array, so if you know you only have one, you can stop it with:
	tracks[0].stop();
}

function getLocation() {
	if (navigator.geolocation) {
		navigator.geolocation.getCurrentPosition(redirectToPosition);
	} else {
		x.innerHTML = "Geolocation is not supported by this browser.";
	}
}
function redirectToPosition(position) {
  	pos = position;
}

function saveAttendance() {
	  	
}


