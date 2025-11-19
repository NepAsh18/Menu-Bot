import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import re
import time
import requests
import json
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class LLMIntegration:
    """Integration with free LLM APIs"""
    
    def __init__(self):
        self.available_models = {
            'huggingface': 'https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium',
            'ollama': 'http://localhost:11434/api/generate',  # Local Ollama
            'together': 'https://api.together.xyz/v1/chat/completions',  # Free tier available
        }
        
    def query_huggingface(self, prompt: str, context: str = "") -> str:
        """Query Hugging Face inference API"""
        try:
            headers = {
                "Authorization": f"Bearer {os.getenv('HF_API_KEY', '')}",
                "Content-Type": "application/json"
            }
            
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
            
            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.7,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                self.available_models['huggingface'],
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', 'Sorry, I could not generate a response.')
                return str(result)
            else:
                logger.warning(f"Hugging Face API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Hugging Face query failed: {e}")
            return None
    
    def query_ollama(self, prompt: str, context: str = "", model: str = "llama2") -> str:
        """Query local Ollama instance"""
        try:
            payload = {
                "model": model,
                "prompt": f"Based on this menu context: {context}\n\nAnswer this question: {prompt}",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 150
                }
            }
            
            response = requests.post(
                self.available_models['ollama'],
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Sorry, I could not generate a response.')
            else:
                logger.warning(f"Ollama API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Ollama query failed: {e}")
            return None
    
    def smart_llm_response(self, prompt: str, context: str = "") -> str:
        """Smart LLM response with fallback mechanisms"""
        # Try Ollama first (local, free)
        response = self.query_ollama(prompt, context)
        if response and len(response) > 10:
            return response
        
        # Fallback to Hugging Face
        response = self.query_huggingface(prompt, context)
        if response and len(response) > 10:
            return response
        
        # Final fallback - rule-based response
        return self.rule_based_fallback(prompt, context)
    
    def rule_based_fallback(self, prompt: str, context: str) -> str:
        """Rule-based fallback when LLMs are unavailable"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['price', 'cost', 'how much']):
            return "I can help you find price information from our menu. Let me search for that."
        elif any(word in prompt_lower for word in ['ingredient', 'contain', 'made of']):
            return "I'll check the ingredients for you in our menu database."
        elif any(word in prompt_lower for word in ['recommend', 'suggest']):
            return "Based on our menu, I can recommend some great options for you."
        else:
            return "I can help you search our menu for that information."

class EnhancedMenuQnA:
    def __init__(self):
        self.model = None
        self.index = None
        self.menu_items = None
        self.llm = LLMIntegration()
        self.conversation_history = []
        self.load_resources()
    
    def load_resources(self):
        """Load the model, index, and metadata"""
        INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index/menu_index.faiss")
        META_PATH = os.getenv("META_PATH", "faiss_index/menu_metadata.pkl")
        MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        
        try:
            logger.info("🔄 Loading AI resources...")
            start_time = time.time()
            
            self.model = SentenceTransformer(MODEL_NAME)
            self.index = faiss.read_index(INDEX_PATH)
            
            with open(META_PATH, "rb") as f:
                self.menu_items = pickle.load(f)
            
            load_time = time.time() - start_time
            logger.info(f"✅ Loaded {len(self.menu_items)} menu items in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error loading resources: {e}")
            raise
    
    def semantic_search(self, query: str, filters: Dict = None, top_k: int = 5) -> List[Dict]:
        """Enhanced semantic search with hybrid ranking"""
        if filters is None:
            filters = {}
        
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
        
        # Search in vector space
        distances, indices = self.index.search(query_embedding, min(50, len(self.menu_items)))
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx >= len(self.menu_items):
                continue
            
            item = self.menu_items[idx]
            
            # Apply filters
            if not self._passes_filters(item, filters):
                continue
            
            # Calculate enhanced score
            score = self._calculate_enhanced_score(item, query, distance, filters)
            
            results.append({
                **item,
                'score': score,
                'semantic_similarity': float(1 / (1 + distance)),
                'search_time': time.time() - start_time
            })
        
        # Sort by combined score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _passes_filters(self, item: Dict, filters: Dict) -> bool:
        """Check if item passes all filters"""
        if filters.get('category') and item['category'].lower() != filters['category'].lower():
            return False
        
        if filters.get('max_price') and item['price'] > filters['max_price']:
            return False
        
        if filters.get('min_price') and item['price'] < filters['min_price']:
            return False
        
        if filters.get('ingredients_include'):
            ingredients_lower = item['ingredients'].lower()
            if not all(ingredient in ingredients_lower for ingredient in filters['ingredients_include']):
                return False
        
        if filters.get('ingredients_exclude'):
            ingredients_lower = item['ingredients'].lower()
            if any(ingredient in ingredients_lower for ingredient in filters['ingredients_exclude']):
                return False
        
        return True
    
    def _calculate_enhanced_score(self, item: Dict, query: str, distance: float, filters: Dict) -> float:
        """Calculate enhanced relevance score"""
        base_score = 1 / (1 + distance)
        
        # Keyword boost
        query_terms = set(re.findall(r'\w+', query.lower()))
        item_text = f"{item['name']} {item['description']} {item['category']} {item['ingredients']}".lower()
        
        keyword_boost = 0
        for term in query_terms:
            if len(term) > 2 and term in item_text:
                keyword_boost += 0.05
        
        # Filter alignment boost
        filter_boost = 0
        if filters.get('category') and item['category'].lower() == filters['category'].lower():
            filter_boost += 0.1
        
        return min(base_score + keyword_boost + filter_boost, 1.0)
    
    def intelligent_intent_parsing(self, question: str) -> Dict[str, Any]:
        """Advanced intent parsing with context awareness"""
        question_lower = question.lower()
        
        # Initialize intent structure
        intent = {
            'type': 'search',
            'filters': {},
            'entities': [],
            'confidence': 0.8,
            'requires_llm': False
        }
        
        # Entity extraction
        self._extract_entities(question_lower, intent)
        
        # Intent classification
        self._classify_intent(question_lower, intent)
        
        # Context integration from conversation history
        self._integrate_conversation_context(intent)
        
        return intent
    
    def _extract_entities(self, question: str, intent: Dict):
        """Extract entities from question"""
        # Price entities
        price_matches = re.findall(r'\$?(\d+\.?\d*)', question)
        if price_matches:
            numbers = [float(match) for match in price_matches if match]
            if 'under' in question or 'less than' in question:
                intent['filters']['max_price'] = max(numbers) if numbers else 10.0
            elif 'over' in question or 'more than' in question:
                intent['filters']['min_price'] = min(numbers) if numbers else 15.0
        
        # Category entities
        category_keywords = {
            'dessert': ['dessert', 'sweet', 'cake', 'pastry', 'ice cream'],
            'main course': ['main', 'entree', 'dinner', 'meal', 'burger', 'pizza'],
            'appetizer': ['appetizer', 'starter', 'salad', 'soup'],
            'side': ['side', 'fries', 'potato']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in question for keyword in keywords):
                intent['filters']['category'] = category
                break
        
        # Ingredient entities
        if 'with' in question:
            match = re.search(r'with\s+([^.?!]*)', question)
            if match:
                ingredients = [word for word in re.findall(r'\w+', match.group(1)) if len(word) > 3]
                intent['filters']['ingredients_include'] = ingredients
        
        if 'without' in question:
            match = re.search(r'without\s+([^.?!]*)', question)
            if match:
                ingredients = [word for word in re.findall(r'\w+', match.group(1)) if len(word) > 3]
                intent['filters']['ingredients_exclude'] = ingredients
    
    def _classify_intent(self, question: str, intent: Dict):
        """Classify user intent"""
        if any(word in question for word in ['compare', 'vs', 'versus', 'difference']):
            intent['type'] = 'comparison'
            intent['requires_llm'] = True
        elif any(word in question for word in ['recommend', 'suggest', 'should i', 'what should']):
            intent['type'] = 'recommendation'
            intent['requires_llm'] = True
        elif any(word in question for word in ['why', 'how', 'explain', 'tell me about']):
            intent['type'] = 'explanation'
            intent['requires_llm'] = True
        elif any(word in question for word in ['similar to', 'like', 'same as']):
            intent['type'] = 'similarity'
        else:
            intent['type'] = 'search'
    
    def _integrate_conversation_context(self, intent: Dict):
        """Integrate context from conversation history"""
        if len(self.conversation_history) > 0:
            last_interaction = self.conversation_history[-1]
            # Carry over filters from previous conversation if relevant
            if 'filters' in last_interaction and intent['type'] == 'search':
                intent['filters'].update(last_interaction['filters'])
    
    def generate_llm_enhanced_response(self, question: str, search_results: List[Dict], intent: Dict) -> str:
        """Generate LLM-enhanced response"""
        # Prepare context for LLM
        context = self._prepare_llm_context(search_results, intent)
        
        # Determine if we need LLM
        if intent['requires_llm'] or len(search_results) > 3 or intent['type'] in ['comparison', 'explanation']:
            llm_response = self.llm.smart_llm_response(question, context)
            
            if llm_response and len(llm_response) > 20:  # Valid LLM response
                return self._format_llm_response(llm_response, search_results)
        
        # Fallback to structured response
        return self._generate_structured_response(question, search_results, intent)
    
    def _prepare_llm_context(self, results: List[Dict], intent: Dict) -> str:
        """Prepare context for LLM"""
        context = f"Menu Items Information:\n\n"
        
        for i, item in enumerate(results[:5]):  # Limit context to top 5 items
            context += f"Item {i+1}:\n"
            context += f"Name: {item['name']}\n"
            context += f"Description: {item['description']}\n"
            context += f"Category: {item['category']}\n"
            context += f"Ingredients: {item['ingredients']}\n"
            context += f"Price: ${item['price']:.2f}\n"
            context += f"Relevance Score: {item['score']:.3f}\n\n"
        
        context += f"User Intent: {intent['type']}\n"
        if intent['filters']:
            context += f"Filters: {intent['filters']}\n"
        
        return context
    
    def _format_llm_response(self, llm_response: str, search_results: List[Dict]) -> str:
        """Format LLM response with structured data"""
        response = f"{llm_response}\n\n"
        
        # Add quick reference to top results
        if search_results:
            response += "🍽️ **Quick Reference:**\n"
            for i, item in enumerate(search_results[:3]):
                response += f"{i+1}. {item['name']} - ${item['price']:.2f} ({item['score']:.0%} match)\n"
        
        return response
    
    def _generate_structured_response(self, question: str, results: List[Dict], intent: Dict) -> str:
        """Generate structured response without LLM"""
        if not results:
            return self._generate_no_results_response(question)
        
        if intent['type'] == 'comparison' and len(results) >= 2:
            return self._generate_comparison_response(results[:2])
        elif intent['type'] == 'recommendation':
            return self._generate_recommendation_response(results)
        else:
            return self._generate_search_response(results, intent)
    
    def _generate_search_response(self, results: List[Dict], intent: Dict) -> str:
        """Generate search results response"""
        response = "🔍 **Here's what I found:**\n\n"
        
        for i, item in enumerate(results[:3]):
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            response += f"{emoji} **{item['name']}**\n"
            response += f"   💰 ${item['price']:.2f} | 🏷️ {item['category']}\n"
            response += f"   📝 {item['description']}\n"
            
            if intent['type'] == 'ingredient_inquiry':
                response += f"   🛒 Ingredients: {item['ingredients']}\n"
            
            response += f"   ✅ Match: {item['score']:.0%}\n\n"
        
        if len(results) > 3:
            response += f"*+ {len(results) - 3} more items found*"
        
        return response
    
    def _generate_comparison_response(self, items: List[Dict]) -> str:
        """Generate comparison response"""
        if len(items) < 2:
            return "I need at least two items to compare."
        
        item1, item2 = items[0], items[1]
        
        response = "⚖️ **Comparison:**\n\n"
        
        # Price comparison
        price_diff = abs(item1['price'] - item2['price'])
        if item1['price'] < item2['price']:
            price_text = f"{item1['name']} is ${price_diff:.2f} cheaper"
        else:
            price_text = f"{item2['name']} is ${price_diff:.2f} cheaper"
        
        # Ingredient overlap
        ingredients1 = set(item1['ingredients'].lower().split(', '))
        ingredients2 = set(item2['ingredients'].lower().split(', '))
        common_ingredients = ingredients1.intersection(ingredients2)
        
        response += f"**{item1['name']}** vs **{item2['name']}**\n\n"
        response += f"💰 **Price**: {price_text}\n"
        response += f"🏷️ **Categories**: Both are {item1['category']}\n"
        response += f"🛒 **Common Ingredients**: {', '.join(common_ingredients)[:100]}...\n\n"
        
        response += f"🥇 **Best Match**: {item1['name']} ({item1['score']:.0%})"
        
        return response
    
    def _generate_recommendation_response(self, results: List[Dict]) -> str:
        """Generate recommendation response"""
        if not results:
            return "I couldn't find any recommendations based on your preferences."
        
        top_item = results[0]
        
        response = "🌟 **My Top Recommendation**\n\n"
        response += f"**{top_item['name']}** - ${top_item['price']:.2f}\n"
        response += f"*{top_item['description']}*\n\n"
        response += f"**Why I recommend this:**\n"
        response += f"• Perfect match for your request ({top_item['score']:.0%})\n"
        response += f"• Great value at ${top_item['price']:.2f}\n"
        response += f"• Contains: {top_item['ingredients']}\n\n"
        
        if len(results) > 1:
            response += "💡 **Other great options:**\n"
            for item in results[1:4]:
                response += f"• {item['name']} - ${item['price']:.2f}\n"
        
        return response
    
    def _generate_no_results_response(self, question: str) -> str:
        """Generate response when no results found"""
        suggestions = [
            "Try using different keywords or being more specific",
            "You can ask about specific categories like 'desserts' or 'main courses'",
            "Try searching by ingredients or price range",
            "Ask me to recommend something based on your preferences"
        ]
        
        response = "🤔 I couldn't find any exact matches for your query.\n\n"
        response += "💡 **Suggestions:**\n"
        for suggestion in suggestions[:2]:
            response += f"• {suggestion}\n"
        
        return response
    
    def ask_question(self, question: str) -> str:
        """Main Q&A interface"""
        logger.info(f"User question: {question}")
        
        # Parse intent
        start_time = time.time()
        intent = self.intelligent_intent_parsing(question)
        
        # Perform semantic search
        search_query = self._extract_search_query(question, intent)
        search_results = self.semantic_search(search_query, intent['filters'])
        
        # Generate response
        response = self.generate_llm_enhanced_response(question, search_results, intent)
        
        # Update conversation history
        self.conversation_history.append({
            'question': question,
            'intent': intent,
            'results_count': len(search_results),
            'timestamp': time.time()
        })
        
        # Keep only last 10 interactions
        self.conversation_history = self.conversation_history[-10:]
        
        total_time = time.time() - start_time
        logger.info(f"Response generated in {total_time:.2f}s")
        
        return response
    
    def _extract_search_query(self, question: str, intent: Dict) -> str:
        """Extract search query from question"""
        # Remove filter-related phrases for cleaner semantic search
        query = question.lower()
        
        # Remove price phrases
        query = re.sub(r'\$?\d+\.?\d*', '', query)
        query = re.sub(r'(under|over|less than|more than)\s*\$\d+', '', query)
        
        # Remove filter words
        filter_words = ['with', 'without', 'category', 'price', 'under', 'over']
        for word in filter_words:
            query = query.replace(word, '')
        
        return query.strip() or question

def interactive_chat():
    """Interactive chat interface"""
    print("🚀 Initializing Advanced Menu AI...")
    
    try:
        qna_system = EnhancedMenuQnA()
        print("✅ System ready!")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    print("\n" + "="*70)
    print("🍽️  ADVANCED MENU AI CHAT")
    print("="*70)
    print("💬 I can help you with:")
    print("   • Finding menu items by description, ingredients, or price")
    print("   • Comparing different menu items")
    print("   • Recommending dishes based on your preferences")
    print("   • Answering detailed questions about ingredients and preparation")
    print("   • Finding similar items to ones you like")
    print("\n🔥 Try these examples:")
    print("   • 'Recommend a dessert under $5 with chocolate'")
    print("   • 'Compare the burger and pizza'")
    print("   • 'What are your vegetarian options?'")
    print("   • 'Find items similar to the cupcake'")
    print("   • 'What ingredients are in the main courses?'")
    print("="*70)
    
    while True:
        try:
            user_input = input("\n🎤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("👋 Thank you for chatting! Have a great day! 🍕")
                break
            
            if not user_input:
                continue
            
            print("\n🤖 AI: ", end="", flush=True)
            
            # Stream the response for better UX
            response = qna_system.ask_question(user_input)
            
            # Print response with slight delay for natural feel
            for line in response.split('\n'):
                print(line)
                time.sleep(0.1)
            
        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using Menu AI! Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Sorry, I encountered an error: {e}")
            print("Please try rephrasing your question.")

def performance_demo():
    """Demonstrate system performance"""
    print("🎯 Advanced Menu AI Performance Demo\n")
    
    qna_system = EnhancedMenuQnA()
    
    demo_questions = [
        "What desserts do you have under $5?",
        "Compare the chocolate cake and ice cream",
        "Recommend a main course with chicken that's affordable",
        "What are the ingredients in your burgers?",
        "Find me vegetarian options without dairy",
        "What's similar to the margherita pizza?",
        "Show me all items with cheese under $10",
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"{i}. ❓ {question}")
        start_time = time.time()
        response = qna_system.ask_question(question)
        response_time = time.time() - start_time
        
        print(f"🤖 {response}")
        print(f"⚡ Response time: {response_time:.2f}s")
        print("-" * 80 + "\n")

if __name__ == "__main__":
    # Run performance demo
    performance_demo()
    
    # Start interactive chat
    interactive_chat()