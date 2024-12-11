// Import delle librerie necessarie
import "cheerio"; // Parser HTML per analizzare documenti web
import 'dotenv/config'; // Per caricare variabili d'ambiente da un file .env

// Import di utilità e moduli da LangChain
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"; // Per dividere i documenti in parti
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio"; // Loader per caricare documenti web
import { OllamaEmbeddings } from "@langchain/ollama"; // Modello per generare embeddings
import { MemoryVectorStore } from "langchain/vectorstores/memory"; // Vector store in memoria
import { ChatOllama } from "@langchain/ollama"; // Modello di chat basato su Ollama
import { StringOutputParser } from "@langchain/core/output_parsers"; // Parser per i risultati in formato stringa
import { PromptTemplate } from "@langchain/core/prompts"; // Per creare template di prompt
import { createStuffDocumentsChain } from "langchain/chains/combine_documents"; // Catena per combinare documenti

// Passo 1: Caricare documenti da remoto
console.log("Carico documenti...");

// URL del sito web da analizzare
const loader = new CheerioWebBaseLoader(
    "https://morpurgo.di.unimi.it/didattica/LabProgrammazione/i_promessi_sposi.txt"
);
const docs = await loader.load(); // Caricamento dei documenti
console.log("Documenti caricati:", docs.length);

// Passo 2: Dividere i documenti in chunk
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000, // Dimensione di ogni chunk
    chunkOverlap: 0, // Sovrapposizione tra i chunk
});
const allSplits = await textSplitter.splitDocuments(docs); // Creazione dei chunk
console.log("Numero di chunks:", allSplits.length);

// Passo 3: Creare il vector store con embeddings
const embeddings = new OllamaEmbeddings({ 
    model: process.env.OLLAMA_EMBEDDING_MODEL // Modello per calcolare embeddings
});
const vectorStore = await MemoryVectorStore.fromDocuments(
    allSplits, // Chunks dei documenti
    embeddings // Modello embeddings
);

// Passo 4: Configurare il modello LLM per il riassunto
const ollamaLlm = new ChatOllama({
    baseUrl: process.env.OLLAMA_URL, // URL del server Ollama
    model: process.env.OLLAMA_MODEL, // Nome del modello
});

// Creare un template per il prompt
const prompt = PromptTemplate.fromTemplate(
    "Riassumi i temi principali dei seguenti documenti: {context}" // Template del prompt
);

// Creare la catena di riassunto dei documenti
const chain = await createStuffDocumentsChain({
    llm: ollamaLlm, // Modello LLM
    outputParser: new StringOutputParser(), // Parser per l'output
    prompt, // Template del prompt
});

// Passo 5: Cercare documenti rilevanti
console.log("Ricerca...");

const question = "Che tipo di relazione intercorre tra renzo e lucia? Queale è il loro rapporto?"; // Domanda specifica
const rd = await vectorStore.similaritySearch(question, 5); // Cerca i 5 documenti più simili
console.log("Risultati trovati:", rd.length);

// Stampare i documenti rilevanti
rd.forEach(r => console.log(r.pageContent, "\n----"));

// Passo 6: Riassumere i risultati
console.log("Summarization...");

const result = await chain.invoke({
    context: rd, // Fornisci i documenti trovati come contesto
});

// Stampare il risultato del riassunto
console.log("\n==== Riassunto ====\n", result, "\n====");
