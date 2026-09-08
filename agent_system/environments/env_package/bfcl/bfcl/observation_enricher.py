import inspect
from typing import Any, Dict, List

def gorilla_fs_enricher(m_name: str, res: Any, instance: Any) -> Any:
    """
    Gorilla File System 结果增强器：
    1. 将 None 返回值转化为明确的 success 状态。
    2. 为 cd/mkdir/touch/mv 等操作自动添加路径和目录快照。
    3. 针对错误自动提供排查建议（ls）。
    """
    if res is None:
        res = {
            "observation": f"Command '{m_name}' completed successfully."
        }
    
    if not isinstance(res, dict):
        return res

    try:
        res["_current_abs_path"] = instance.pwd().get("current_working_directory", "/")
    except:
        pass

    state_changing_methods = ["cd", "mkdir", "touch", "rm", "rmdir", "mv", "cp", "echo"]
    if m_name in state_changing_methods and "error" not in res:
        try:
            res["_new_directory_contents"] = instance.ls().get("current_directory_content", [])
        except:
            pass

    return res


def vehicle_control_enricher(m_name: str, res: Any, instance: Any) -> Any:
    print("Inside vehicle_control_enricher")
    """Vehicle Control: 动作执行后自动反馈车辆状态"""
    if not isinstance(res, dict): return res
    if m_name == "startEngine" and "error" in res:
        res["_ignition_diagnostic"] = {
            "all_doors_locked": instance.remainingUnlockedDoors == 0,
            "unlocked_list": [d for d, s in instance.doorStatus.items() if s == "unlocked"],
            "brake_pedal_position": instance._brakePedalForce / 1000.0, # 转化为 0-1
            "fuel_level_gallons": instance.fuelLevel,
            "battery_voltage": instance.batteryVoltage
        }
        res["_hint"] = "Ensure doors are locked, brake is pressed to 1.0, and fuel is > 0."
    

    if m_name == "check_tire_pressure":
        res["_decision_safety_hint"] = (
            "NOTICE: 'healthy_tire_pressure' is based on the factory standard (30-35 psi). "
        )

    if m_name in ["lockDoors", "unlockDoors"]:
        res["_current_door_grid"] = instance.doorStatus
        res["_summary"] = f"{instance.remainingUnlockedDoors} doors currently unlocked."

    return res


def trading_bot_enricher(m_name: str, res: Any, instance: Any) -> Any:
    """
    Trading Bot 结果增强器：
    1. 自动处理认证失败，提供引导。
    2. 针对交易操作，进行价格偏移校验和余额对账。
    3. 明确账户数据范围，防止模型对个人信息（Email/Phone）产生幻觉。
    4. 增强市场时间感知。
    """
    if res is None:
        return {"status": "success", "message": f"Operation {m_name} executed."}
    
    if not isinstance(res, dict):
        return res

    if "error" in res and ("not authenticated" in res["error"].lower() or "log in" in res["error"].lower()):
        res["_action_required"] = "Please call 'trading_login(username, password)' to enable this feature."
        return res

    if m_name == "place_order" and "error" not in res:
        symbol = res.get("symbol")
        ordered_price = float(res.get("price", 0))
        amount = int(res.get("amount", 0))
        
        if symbol in instance.stocks:
            current_market_price = instance.stocks[symbol]["price"]
            price_diff_percent = abs(ordered_price - current_market_price) / current_market_price
            
            res["_market_verification"] = {
                "current_market_price": current_market_price,
                "price_deviation_percent": f"{price_diff_percent:.2%}",
                "is_stale_price": price_diff_percent > 0.05  # 偏差超过5%定义为陈旧价格
            }
            
            total_cost = ordered_price * amount
            res["_financial_impact"] = {
                "estimated_total_cost": total_cost,
                "remaining_balance_after": instance.account_info["balance"] - total_cost
            }


    time_sensitive_ops = ["place_order", "withdraw_funds", "update_market_status"]
    if m_name in time_sensitive_ops or "market_status" in m_name:
        res["_market_context"] = {
            "current_system_time": instance.get_current_time()["current_time"],
            "market_status": instance.market_status,
            "trading_hours": "09:30 AM - 04:00 PM"
        }
        if instance.market_status == "Closed":
            res["_market_warning"] = "Market is currently CLOSED. Some operations like withdrawals may fail."

    if m_name == "get_symbol_by_name" and res.get("symbol") == "Stock not found":
        res["_search_tips"] = "Try 'get_available_stocks(sector)' to browse valid symbols by industry (e.g., 'Technology')."

    if m_name == "get_order_details" and "status" in res:
        status_map = {
            "Pending": "Order placed but not yet on the exchange book.",
            "Open": "Order is live on the market and can be cancelled.",
            "Completed": "Order fully executed. Cannot be modified or cancelled."
        }
        res["_status_description"] = status_map.get(res["status"], "")

    return res

def travel_booking_enricher(m_name: str, res: Any, instance: Any) -> Any:
    """
    Travel API 增强器：
    1. 监控 Token 剩余次数，防止调用中断。
    2. 自动进行“余额+预算”双重校验预警。
    3. 针对身份核验失败，带出具体的系统规则。
    4. 针对航线/城市输入错误，提供 IATA 代码自动校对。
    """
    if not isinstance(res, dict): return res

    if instance.access_token:
        res["_token_status"] = {
            "remaining_calls": instance.token_expires_in,
            "status": "Healthy" if instance.token_expires_in > 1 else "Critical (Refresh Required Soon)"
        }
        if instance.token_expires_in <= 1:
            res["_token_hint"] = "Token is about to expire. You may need to call 'authenticate_travel' again after this or the next step."

    if m_name == "verify_traveler_information":
        if res.get("verification_status") is False:
            res["_validation_rules_reference"] = {
                "min_age": 18,
                "required_passport_prefix": "US",
                "identity_match": f"Must match registered user: {instance.user_first_name} {instance.user_last_name}"
            }

    if m_name == "get_flight_cost" and "travel_cost_list" in res:
        costs = res["travel_cost_list"]
        if costs:
            min_cost = min(costs)
            max_balance = max([card["balance"] for card in instance.credit_card_list.values()]) if instance.credit_card_list else 0
            
            res["_financial_feasibility"] = {
                "can_afford_any": max_balance >= min_cost,
                "current_budget_limit": instance.budget_limit,
                "note": f"Booking will fail if (Card Balance - Cost) < Budget Limit ({instance.budget_limit})."
            }

    if m_name == "book_flight" and res.get("booking_status") is False:
        error = res.get("error", "")
        if "Invalid" in error and ("airport" in error or "location" in error):
            res["_iata_suggestion_hint"] = "Use 'list_all_airports' or 'get_nearest_airport_by_city' to confirm IATA codes."
        
        if "funds" in error or "budget" in error:
            res["_account_snapshot"] = {
                "cards": {cid: f"Balance: {c['balance']}" for cid, c in instance.credit_card_list.items()},
                "active_budget_limit": instance.budget_limit
            }


    return res



ENRICHMENT_HANDLERS = {
    "GorillaFileSystem": gorilla_fs_enricher,
    "VehicleControlAPI": vehicle_control_enricher,
    "TradingBot": trading_bot_enricher,
    "TravelAPI": travel_booking_enricher,
}
