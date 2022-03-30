import IntegrationAtRisk
import IntegrationDefiniteCustomer

suggestion_list=IntegrationAtRisk.run_recommendation_for_input('A1055QXUA6BOEL','B00000I1C0')
high_price_list=IntegrationDefiniteCustomer.run_recommendation_for_input('A1055QXUA6BOEL','B00000I1C0')

print(suggestion_list)
print(type(suggestion_list))

print(high_price_list)
print(type(high_price_list))