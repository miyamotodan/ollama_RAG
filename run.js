// Import delle librerie necessarie
import "cheerio"; // Parser HTML per analizzare documenti web
import 'dotenv/config'; // Per caricare variabili d'ambiente da un file .env

// Import di utilità e moduli da LangChain
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"; // Per dividere i documenti in parti
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio"; // Loader per caricare documenti web
import { DocxLoader } from "@langchain/community/document_loaders/fs/docx";
import { OllamaEmbeddings } from "@langchain/ollama"; // Modello per generare embeddings
import { MemoryVectorStore } from "langchain/vectorstores/memory"; // Vector store in memoria
import { ChatOllama } from "@langchain/ollama"; // Modello di chat basato su Ollama
import { StringOutputParser } from "@langchain/core/output_parsers"; // Parser per i risultati in formato stringa
import { PromptTemplate } from "@langchain/core/prompts"; // Per creare template di prompt
import { createStuffDocumentsChain } from "langchain/chains/combine_documents"; // Catena per combinare documenti

const getTimeBetweenDates = (startDate, endDate) => {
    const seconds = Math.floor((endDate - startDate) / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    return { seconds, minutes, hours, days };
};

// Passo 1: Caricare documenti da remoto
console.log("Carico documenti...");

// URL del sito web da analizzare
//const loader = new CheerioWebBaseLoader(
//    "https://www.forumpa.it/pa-digitale/gestione-documentale/polo-di-conservazione-digitale-la-sfida-dellarchivio-centrale-dello-stato-per-un-nuovo-modello-conservativo/"
//);

const loader = new DocxLoader("250624_PSC_MdL.docx");

let d1 = new Date().getTime();
const docs = await loader.load(); // Caricamento dei documenti
let d2 = new Date().getTime();
console.log(getTimeBetweenDates(d1, d2));

console.log("Documenti caricati:", docs.length);

// Passo 2: Dividere i documenti in chunk
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500, // Dimensione di ogni chunk
    chunkOverlap: 0, // Sovrapposizione tra i chunk
});

d1 = new Date().getTime();
const allSplits = await textSplitter.splitDocuments(docs); // Creazione dei chunk
d2 = new Date().getTime();
console.log(getTimeBetweenDates(d1, d2));

console.log("Numero di chunks:", allSplits.length);

// Passo 3: Creare il vector store con embeddings
const embeddings = new OllamaEmbeddings({
    model: process.env.OLLAMA_EMBEDDING_MODEL // Modello per calcolare embeddings
});

d1 = new Date().getTime();
const vectorStore = await MemoryVectorStore.fromDocuments(
    allSplits, // Chunks dei documenti
    embeddings // Modello embeddings
);
d2 = new Date().getTime();
console.log(getTimeBetweenDates(d1, d2));


// Passo 4: Configurare il modello LLM per il riassunto
const ollamaLlm = new ChatOllama({
    baseUrl: process.env.OLLAMA_URL, // URL del server Ollama
    model: process.env.OLLAMA_MODEL, // Nome del modello
});

// Creare un template per il prompt
const prompt = PromptTemplate.fromTemplate(
    "Riassumi brevemente in massimo 200 parole il contenuto dei seguenti documenti: {context}" // Template del prompt
);

// Creare la catena di riassunto dei documenti
const chain = await createStuffDocumentsChain({
    llm: ollamaLlm, // Modello LLM
    outputParser: new StringOutputParser(), // Parser per l'output
    prompt, // Template del prompt
});

// Passo 5: Cercare documenti rilevanti
console.log("Ricerca...");

const question = "fai un elenco delle principali mansioni del 'direttore lavori'"; // Domanda specifica

d1 = new Date().getTime();
const rd = await vectorStore.similaritySearch(question, 10); // Cerca i 5 documenti più simili
d2 = new Date().getTime();
console.log(getTimeBetweenDates(d1, d2));

console.log("Risultati trovati:", rd.length);

// Stampare i documenti rilevanti
rd.forEach(r => console.log(r.pageContent, "\n----"));

// Passo 6: Riassumere i risultati
console.log("Summarization...");

d1 = new Date().getTime();
const result = await chain.invoke({
    context: rd, // Fornisci i documenti trovati come contesto
});
d2 = new Date().getTime();
console.log(getTimeBetweenDates(d1, d2));

// Stampare il risultato del riassunto
console.log("\n==== Riassunto ====\n", result, "\n====");


