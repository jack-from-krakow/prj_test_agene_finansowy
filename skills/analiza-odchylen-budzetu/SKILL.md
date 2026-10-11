---
name: analiza-odchylen-budzetu
description: Analiza odchyleń wykonania od planu budżetu (plan vs wykonanie) w podziale na kategorie i miesiące. Użyj, gdy użytkownik pyta o przekroczenia budżetu, odchylenia albo realizację planu.
---

# Analiza odchyleń budżetu

1. Ustal okres (miesiąc, kwartał, rok narastająco) i zakres (całość, dział, kategoria).
2. `describe_table` dla arkusza z planem i wykonaniem, potwierdź nazwy kolumn.
3. `aggregate_table`: suma `Plan` i `Wykonanie` w podziale na wybrany wymiar, z filtrem okresu.
4. Dla każdej pozycji policz `calculate`: odchylenie = Wykonanie − Plan,
   odchylenie % = (Wykonanie − Plan) / Plan × 100.
5. Oznacz pozycje przekraczające progi z polityki kosztowej (>10% wymaga wyjaśnienia,
   >20% raport do zarządu). Jeśli polityka nie jest znana, zapytaj orkiestratora/RAG albo podaj założenie.
6. Sprawdź `/finance/` pod kątem notatek, które tłumaczą odchylenia (np. planowane przesunięcia).
7. Wynik: tabela (pozycja, plan, wykonanie, odchylenie, odchylenie %), 3–5 wniosków, źródła.
   Zapisz tabelę w `/workspace/` jeśli użytkownik będzie do niej wracał.
