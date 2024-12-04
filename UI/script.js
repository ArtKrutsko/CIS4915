document.addEventListener("DOMContentLoaded", () => {
    const tableBody = document.querySelector("#data-table tbody");
    const predictButton = document.getElementById("predict-button");
    const outputDiv = document.getElementById("output");
    const outputContainer = document.getElementById("output-container");
    const optionSelect = document.getElementById("option-select");
    const customOptionInput = document.getElementById("custom-option");
    const highschoolGPAInput = document.getElementById("highschool-gpa");
    const satScoreInput = document.getElementById("sat-score");
    const predictionToggle = document.getElementById("prediction-toggle");
    const toggleLabel = document.getElementById("toggle-label");

    // Toggle label text change
    predictionToggle.addEventListener("change", () => {
        toggleLabel.textContent = predictionToggle.checked ? "Predict Grade" : "Predict Pass/Fail";
    });

    // Handle prediction button click
    predictButton.addEventListener("click", async () => {
        const data = [];
        const rows = tableBody.querySelectorAll("tr");
        rows.forEach(row => {
            const inputs = row.querySelectorAll("input");
            const classValue = inputs[0].value.trim();
            const gradeValue = inputs[1].value.trim();
            if (classValue && gradeValue) {
                data.push({ class: classValue, grade: gradeValue });
            }
        });

        // Get the selected option
        let selectedOption = optionSelect.value;
        if (selectedOption === "Other") {
            selectedOption = customOptionInput.value.trim();
        }

        // Get Highschool GPA, SAT Score, and toggle state
        const highschoolGPA = highschoolGPAInput.value.trim();
        const satScore = satScoreInput.value.trim();
        const predictType = predictionToggle.checked ? "grade" : "pass_fail";
        console.log("predictType = ", predictType)

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    data,
                    option: selectedOption,
                    highschoolGPA,
                    satScore,
                    predictType,
                }),
            });
            const result = await response.json();

            // Populate output container with dynamic content
            outputContainer.style.display = "block";
            outputContainer.innerHTML = ""; // Clear previous content

            // Add text and images dynamically
            result.output.forEach(item => {
                const outputItem = document.createElement("div");
                outputItem.className = "output-item";

                if (item.type === "text") {
                    const textElement = document.createElement("p");
                    textElement.textContent = item.content;
                    outputItem.appendChild(textElement);
                } else if (item.type === "image") {
                    const imgElement = document.createElement("img");
                    imgElement.src = item.content; // URL or base64
                    imgElement.alt = "Output Image";
                    imgElement.className = "output-image";
                    outputItem.appendChild(imgElement);
                }

                outputContainer.appendChild(outputItem);
            });
        } catch (error) {
            console.error("Error during prediction:", error);
            outputDiv.textContent = "Error making prediction!";
        }
    });
});
